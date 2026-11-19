import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

import os
import shutil
import wandb

import sys
import yaml
import csv
import re

import Simulacra as sc



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


def check_signal(run_path):
    snr_regex = re.compile(r"(^.*signal.*\.npy)$")
    matches = snr_regex.search(run_path)
    if matches is not None:
        return str(matches.groups()[0])
    else:
        return None
def check_clnsig(run_path):
    snr_regex = re.compile(r"(^.*clnsig.*\.npy)$")
    matches = snr_regex.search(run_path)
    if matches is not None:
        return str(matches.groups()[0])
    else:
        return None

class NoisyToClean1DDataset(Dataset):
    def __init__(self, noisy, clean, normalize=True):
        self.noisy = noisy.astype(np.float32)
        self.clean = clean.astype(np.float32)
        self.normalize = normalize

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        x = self.noisy[idx]   # (1, 512)
        y = self.clean[idx]   # (1, 512)

        if self.normalize:
            mu = x.mean()
            sd = x.std() + 1e-6
            x = (x - mu) / sd
            y = (y - mu) / sd

        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y).float(),
        }


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_path = sys.argv[1]
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    if len(sys.argv) != 3:
        print("ERROR: invalid number of inputs!")
        print(f"Please enter: python {sys.argv[0]} YAML_path runpath")
        sys.exit(1)

    #dataloader
    target_snr = float(config["dataloader"]["target_snr"])
    pulse_length = int(config["dataloader"]["pulse_length"])
    training_length = int(config["dataloader"]["training_length"])
    test_length = int(config["dataloader"]["test_length"])
    val_length = int(config["dataloader"]["val_length"])
    training_seed = int(config["dataloader"]["training_seed"])
    test_seed = int(config["dataloader"]["test_seed"])
    val_seed = int(config["dataloader"]["val_seed"])
    batch_size = int(config["dataloader"]["batch_size"])
    data_path  = str(config["dataloader"]["dataset_path"])

    #model
    in_ch = int(config["model"]["in_ch"])
    hid = int(config["model"]["hid"])
    z_ch = int(config["model"]["z_ch"])
    n_codes = int(config["model"]["n_codes"])

    #adam
    lr = float(config["adam"]["lr"])
    weight_decay = float(config["adam"]["weight_decay"])

    #lr_scheduler
    mode = str(config["lr_scheduler"]["mode"])
    factor = float(config["lr_scheduler"]["factor"])
    patience = int(config["lr_scheduler"]["patience"])
    min_lr = float(config["lr_scheduler"]["min_lr"])

    #training
    num_epochs = int(config["training"]["num_epochs"])

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="gwyndandy-niu",
        # Set the wandb project where this run will be logged.
        project="VQ-VAE Init Testing",
        # Track hyperparameters and run metadata.
        config=config
    )
    #Set run name
    run_name = f"SNR:{target_snr}-Run:{run.id}"
    run.name = run_name

    filepath = f"{sys.argv[2]}/{run_name}/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    csv_filename = f"{filepath}/run_data.csv"

    shutil.copy(yaml_path, f"{filepath}/{yaml_path}")

    if target_snr != 0:
        print(type(target_snr),target_snr)
        asdf
        train_ds = sc.simulacra_dataset(target_snr, training_length, training_seed, pulse_length)
        val_ds = sc.simulacra_dataset(target_snr, val_length, val_seed, pulse_length)
        test_ds = sc.simulacra_dataset(target_snr, test_length, test_seed, pulse_length)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        clean_paths = [x.name for x in os.scandir(data_path) if check_clnsig(x.name) is not None]
        signal_paths = [x.name for x in os.scandir(data_path) if check_signal(x.name) is not None]
        train_noise = []
        train_signal = []
        for cln,sig in zip(clean_paths,signal_paths):
            a=np.load(sig,allow_pickle=True)
            c=np.load(cln,allow_pickle=True)
            nwf=a.size
            ntcks=len(a[0])-54
            for j in range(nwf-50):
                ya=[]
                yc=[]
                j=0
                for i in range(0,ntcks):
                    lab='tck_'+str(i)
                    ya.append(a[j][lab])
                    yc.append(c[j][lab])
                train_noise.append(ya)
                train_signal.append(yc)
            test_noise = []
            test_signal = []
            for j in range(nwf-50,nwf-1):
                ya=[]
                yc=[]
                j=0
                for i in range(0,ntcks):
                    lab='tck_'+str(i)
                    ya.append(a[j][lab])
                    yc.append(c[j][lab])
            test_noise.append(ya)
            test_signal.append(yc)
        train_noise_c = np.reshape(np.array(train_noise), (-1, 1,512))
        train_signal_c = np.reshape(np.array(train_signal), (-1, 1,512))
        test_noise_c = np.reshape(np.array(test_noise), (-1, 1,512))
        test_signal_c = np.reshape(np.array(test_signal), (-1, 1,512))
        train_ds_full = NoisyToClean1DDataset(
            noisy=train_noise_c,
            clean=train_signal_c,
            normalize=True,
        )

        test_ds = NoisyToClean1DDataset(
            noisy=test_noise_c,
            clean=test_signal_c,
            normalize=True,
        )

        n_val = int(0.1 * len(train_ds_full))
        n_train = len(train_ds_full) - n_val

        train_ds, val_ds = random_split(
            train_ds_full,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model = VQVAE1DDenoiser(in_ch, hid, z_ch, n_codes).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )

    def run_epoch(loader, train=True, lambda_vq=0.25, max_grad_norm=1.0):
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
            "vq": total_vq / n,
            "skipped": skipped,
        }

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "val_recon": [],
        "train_vq": [],
        "val_vq": [],
        "lr": [],
    }

    best_val = float("inf")
    best_state = None
    patience = 15
    bad_epochs = 5

    file = open(csv_filename, mode='w', newline='')
    writer = csv.writer(file)
    step_info = {"Epoch": 0,
                "lr": 0,
                "lambda_vq": 0,
                "train":0,
                "val": 0,
                "recon": 0,
                "vq": 0,
                "skipped": 0}
    writer.writerow(step_info.keys())

    for epoch in range(1, num_epochs + 1):

        # Warm up VQ loss so it does not dominate early training
        lambda_vq = min(0.15, 0.15 * epoch / 10)

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

        scheduler.step(val_stats["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_recon"].append(train_stats["recon"])
        history["val_recon"].append(val_stats["recon"])
        history["train_vq"].append(train_stats["vq"])
        history["val_vq"].append(val_stats["vq"])
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
        step_info = {"Epoch": epoch,
                    "lr": current_lr,
                    "lambda_vq": lambda_vq,
                    "train": train_stats['loss'],
                    "val": val_stats['loss'],
                    "recon": val_stats['recon'],
                    "vq": val_stats['vq'],
                    "skipped": train_stats['skipped']}
        
        writer.writerow(step_info.values())
        run.log(step_info)

        if bad_epochs >= patience:
            print("Early stopping.")
            break
    file.close()

    model.load_state_dict(best_state)
    model.to(device)

    test_stats = run_epoch(test_loader, train=False, lambda_vq=0.25)

    with open(f"{filepath}/test_stats.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(test_stats.keys())  # Write header
        writer.writerow(test_stats.values())  # Write values


    model.eval()

    batch = next(iter(test_loader))

    x = batch["x"].to(device)
    y = batch["y"].to(device)

    with torch.no_grad():
        y_hat, vq_loss, codes = model(x)

    for idx in range(min(5,len(x))):
        noisy = x[idx, 0].cpu().numpy()
        clean = y[idx, 0].cpu().numpy()
        pred = y_hat[idx, 0].cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(clean, label="clean target", linewidth=2)
        ax.plot(noisy, label="noisy input", alpha=0.6)
        ax.plot(pred, label="VQ-VAE output", linewidth=2)

        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        plt.savefig(f"{filepath}/output_{idx}.png")
        run.log({f"output_{idx}": fig})
        plt.close(fig)

    torch.save(model.state_dict(), f"{filepath}/model_dict.pth")
    torch.save(model, f"{filepath}/model.pth")
