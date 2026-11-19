import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label
import yaml

class ResBlock1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))
class Encoder1D(nn.Module):
    def __init__(self, in_ch=1, hid=64, z_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hid, 4, stride=2, padding=1),  # 512 -> 256
            nn.Tanh(),

            nn.Conv1d(hid, hid, 4, stride=2, padding=1),    # 256 -> 128
            nn.Tanh(),

            ResBlock1D(hid),

            nn.Conv1d(hid, z_ch, 1),
        )

    def forward(self, x):
        return self.net(x)
class Decoder1D(nn.Module):
    def __init__(self, out_ch=1, hid=64, z_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(z_ch, hid, 1),

            ResBlock1D(hid),

            nn.ConvTranspose1d(hid, hid, 4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh(),

            nn.ConvTranspose1d(hid, hid, 4, stride=2, padding=1),  # 256 -> 512
            nn.Tanh(),

            nn.Conv1d(hid, out_ch, 3, padding=1),
        )
    def forward(self, z):
        return self.net(z)
class VectorQuantizer1D(nn.Module):
    def __init__(self, n_codes=512, code_dim=64, beta=0.25):
        super().__init__()
        self.beta = beta
        self.codebook = nn.Embedding(n_codes, code_dim)
        self.codebook.weight.data.uniform_(-1 / n_codes, 1 / n_codes)

    def forward(self, z_e):
        # z_e: (B, C, L)
        B, C, L = z_e.shape

        z = z_e.permute(0, 2, 1).contiguous().view(-1, C)

        d = (
            z.pow(2).sum(1, keepdim=True)
            + self.codebook.weight.pow(2).sum(1)
            - 2 * z @ self.codebook.weight.t()
        )

        idx = torch.argmin(d, dim=1)

        z_q = self.codebook(idx)
        z_q = z_q.view(B, L, C).permute(0, 2, 1).contiguous()

        codebook_loss = F.mse_loss(z_q, z_e.detach(),reduction="mean")
        commit_loss = F.mse_loss(z_e, z_q.detach(),reduction="mean")
        vq_loss = codebook_loss + self.beta * commit_loss

        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, idx.view(B, L)
class VQVAE1DDenoiser(nn.Module):
    def __init__(self, in_ch=1, hid=64, z_ch=64, n_codes=512):
        super().__init__()
        self.enc = Encoder1D(in_ch, hid, z_ch)
        self.vq = VectorQuantizer1D(n_codes, z_ch)
        self.dec = Decoder1D(1, hid, z_ch)

    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z_e = self.enc(x)
        z_q, vq_loss, codes = self.vq(z_e)
        clean_hat = self.dec(z_q)
        return clean_hat, vq_loss, codes
def vae_run_epoch(loader, history, train=True, lambda_vq=0.25, max_grad_norm=1.0):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    skipped = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train):
            y_hat, vq_loss, codes = model(x)

            recon_loss = F.huber_loss(y_hat, y, delta=0.5)
            loss =  recon_loss + lambda_vq * vq_loss

            if not torch.isfinite(loss):
                skipped += 1
                continue

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                )

                optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_vq += vq_loss.item()

    n = max(1, len(loader) - skipped)

    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "skipped": skipped,
    }
    
def vae_train(model,history,path):
    for epoch in range(1, num_epochs + 1):

        scheduler.step(val_stats["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        train_stats = run_epoch(
            train_loader,
            train=True,
            lambda_vq=lambda_vq,
            max_grad_norm=1.0,
        )

        val_stats = run_epoch(
            val_loader,
            train=False,
            lambda_vq=lambda_vq,
            max_grad_norm=1.0,
        )

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_recon"].append(train_stats["recon"])
        history["val_recon"].append(val_stats["recon"])
        history["lr"].append(current_lr)

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | "
            f"lr {current_lr:.2e} | "
            f"lambda_vq {lambda_vq:.3f} | "
            f"train {train_stats['loss']:.6f} | "
            f"val {val_stats['loss']:.6f} | "
            f"recon {val_stats['recon']:.6f} | "
            f"vq {val_stats['vq']:.6f} | "
            f"skipped {train_stats['skipped']}"
        )
        
        writer.writerow(history.values())

        if bad_epochs >= patience:
            print("Early stopping.")
            break
    file.close()


def read_yaml_vqvae(yaml_path):
    with open(yaml_path, 'r') as file:
        config_yaml = yaml.safe_load(file)
    config = {}
    #dataloader
    config["target_snr"] = float(config_yaml["dataloader"]["target_snr"])
    config["pulse_length"] = int(config_yaml["dataloader"]["pulse_length"])
    config["training_length"] = int(config_yaml["dataloader"]["training_length"])
    config["test_length"] = int(config_yaml["dataloader"]["test_length"])
    config["val_length"] = int(config_yaml["dataloader"]["val_length"])
    config["training_seed"] = int(config_yaml["dataloader"]["training_seed"])
    config["test_seed"] = int(config_yaml["dataloader"]["test_seed"])
    config["val_seed"] = int(config_yaml["dataloader"]["val_seed"])
    config["batch_size"] = int(config_yaml["dataloader"]["batch_size"])
    config["dataset_path"] = str(config_yaml["dataloader"]["dataset_path"])

    #model
    config["in_ch"] = int(config_yaml["model"]["in_ch"])
    config["hid"] = int(config_yaml["model"]["hid"])
    config["z_ch"] = int(config_yaml["model"]["z_ch"])
    config["n_codes"] = int(config_yaml["model"]["n_codes"])

    #adam
    config["lr"] = float(config_yaml["adam"]["lr"])
    config["weight_decay"] = float(config_yaml["adam"]["weight_decay"])

    #lr_scheduler
    config["mode"] = str(config_yaml["lr_scheduler"]["mode"])
    config["factor"] = float(config_yaml["lr_scheduler"]["factor"])
    config["patience"] = int(config_yaml["lr_scheduler"]["patience"])
    config["min_lr"] = float(config_yaml["lr_scheduler"]["min_lr"])

    #training
    config["num_epochs"] = int(config_yaml["training"]["num_epochs"])
    return config 

def get_vqvae(config):
    model =  VQVAE1DDenoiser(in_ch=config["in_ch"], hid=config["hid"], z_ch=config["z_ch"], n_codes=config["n_codes"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["mode"],
        factor=config["factor"],
        patience=config["patience"],
        min_lr=config["min_lr"],
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "val_recon": [],
        "train_vq": [],
        "val_vq": [],
        "lr": [],
        "skipped": [],
    }
    return model,optimizer,scheduler, history

class autoencoder(nn.Module):
    def __init__(self, hid, z_ch, in_ch, out_ch=200):
        super().__init__()
        mid_ch = np.abs(z_ch-hid)//2 + hid 
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hid, mid_ch, kernel_size = 5, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_ch, z_ch, kernel_size = 9, stride=5, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(z_ch, mid_ch, kernel_size = 9, stride=5, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(mid_ch, hid, kernel_size = 5, stride=4, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hid, in_ch, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(in_ch, in_ch, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.Upsample(size=out_ch, mode='linear', align_corners=False)
        )
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)
def read_yaml_autoencoder(yaml_path):
    with open(yaml_path, 'r') as file:
        config_yaml = yaml.safe_load(file)
    config = {}
    #dataloader
    config["target_snr"] = float(config_yaml["dataloader"]["target_snr"])
    config["pulse_length"] = int(config_yaml["dataloader"]["pulse_length"])
    config["training_length"] = int(config_yaml["dataloader"]["training_length"])
    config["test_length"] = int(config_yaml["dataloader"]["test_length"])
    config["val_length"] = int(config_yaml["dataloader"]["val_length"])
    config["training_seed"] = int(config_yaml["dataloader"]["training_seed"])
    config["test_seed"] = int(config_yaml["dataloader"]["test_seed"])
    config["val_seed"] = int(config_yaml["dataloader"]["val_seed"])
    config["batch_size"] = int(config_yaml["dataloader"]["batch_size"])
    config["dataset_path"] = str(config_yaml["dataloader"]["dataset_path"])

    #model
    config["in_ch"] = int(config_yaml["model"]["in_ch"])
    config["hid"] = int(config_yaml["model"]["hid"])
    config["z_ch"] = int(config_yaml["model"]["z_ch"])

    #adam
    config["lr"] = float(config_yaml["adam"]["lr"])
    config["weight_decay"] = float(config_yaml["adam"]["weight_decay"])

    #lr_scheduler
    config["mode"] = str(config_yaml["lr_scheduler"]["mode"])
    config["factor"] = float(config_yaml["lr_scheduler"]["factor"])
    config["patience"] = int(config_yaml["lr_scheduler"]["patience"])
    config["min_lr"] = float(config_yaml["lr_scheduler"]["min_lr"])

    #training
    config["num_epochs"] = int(config_yaml["training"]["num_epochs"])
    return config 

def get_vae(config):
    model =  autoencoder(in_ch=config["in_ch"], hid=config["hid"], z_ch=config["z_ch"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["mode"],
        factor=config["factor"],
        patience=config["patience"],
        min_lr=config["min_lr"],
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "skipped": [],
    }
    return model,optimizer,scheduler, history