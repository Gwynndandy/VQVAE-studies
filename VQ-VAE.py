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
import wandb

import sys

import Simulacra as sc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    os.chdir("/home/maniacalm/bin/VQVAE_studies")
except FileNotFoundError:
    print("Either filepath is wrong or this is ran on another computer")

#dataloader
target_snr, pulse_length = float(sys.argv[1]),512
training_length,test_length,val_length = 10000,1000,1000
training_seed,test_seed,val_seed = 0,training_length*2,training_length*3
batch_size = 64

#model
in_ch=1
hid=32
z_ch=512
n_codes=32

#adam
lr=1e-4
weight_decay=1e-4

#lr_scheduler
mode="min"
factor=0.5
patience=5
min_lr=1e-5

#training
num_epochs = 100


# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="gwyndandy-niu",
    # Set the wandb project where this run will be logged.
    project="VQ-VAE Init Testing",
    # Track hyperparameters and run metadata.
    config={
        #dataloader
        'target_snr':target_snr,
        'pulse_length':pulse_length,
        'training_length':training_length,
        'test_length':test_length,
        'val_length':val_length,
        'training_seed':training_seed,
        'test_seed':test_seed,
        'val_seed':val_seed,
        'batch_size':batch_size,

        #model
        'in_ch':in_ch,
        'hid':hid,
        'z_ch':z_ch,
        'n_codes':n_codes,

        #adam
        'lr':lr,
        'weight_decay':weight_decay,

        #lr_scheduler
        'mode':mode,
        'factor':factor,
        'patience':patience,
        'min_lr':min_lr,

        #training
        'num_epochs':num_epochs,
    },
)
#Set run name
run.name = f"SNR:{target_snr}-Run:{run.id}"

config = f"""dataloader
        target_snr:{target_snr}
        pulse_length: {pulse_length}
        training_length: {training_length}
        test_length: {test_length}
        val_length: {val_length}
        training_seed: {training_seed}
        test_seed: {test_seed}
        val_seed: {val_seed}
        batch_size: {batch_size}

        model
        in_ch: {in_ch}
        hid: {hid}
        z_ch: {z_ch}
        n_codes: {n_codes}

        adam
        lr: {lr}
        weight_decay: {weight_decay}

        lr_scheduler
        mode: {mode}
        factor: {factor}
        patience: {patience}
        min_lr: {min_lr}

        training
        num_epochs: {num_epochs}"""
print("Config:")
print(config)

train_ds = sc.simulacra_dataset(target_snr, training_length, training_seed, pulse_length)
val_ds = sc.simulacra_dataset(target_snr, val_length, val_seed, pulse_length)
test_ds = sc.simulacra_dataset(target_snr, test_length, test_seed, pulse_length)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

batch = next(iter(train_loader))
batch["x"].shape, batch["y"].shape

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

model = VQVAE1DDenoiser(in_ch, hid, z_ch, n_codes).to(device)

print(model)
batch = next(iter(train_loader))

x = batch["x"].to(device)
y = batch["y"].to(device)

with torch.no_grad():
    y_hat, vq_loss, codes = model(x)

print("x:", x.shape)
print("y:", y.shape)
print("y_hat:", y_hat.shape)
print("codes:", codes.shape)
print("vq_loss:", vq_loss.item())

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
    run.log({"Epoch": epoch,
             "lr": current_lr,
             "lambda_vq": lambda_vq,
             "train": train_stats['loss'],
             "val": val_stats['loss'],
             "recon": val_stats['recon'],
             "vq": val_stats['vq'],
             "skipped": train_stats['skipped']})

    if bad_epochs >= patience:
        print("Early stopping.")
        break

model.load_state_dict(best_state)
model.to(device)

test_stats = run_epoch(test_loader, train=False, lambda_vq=0.25)
print(test_stats)

model.eval()

batch = next(iter(test_loader))

x = batch["x"].to(device)
y = batch["y"].to(device)

with torch.no_grad():
    y_hat, vq_loss, codes = model(x)

for idx in range(5):
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

    run.log({f"output_{idx}": fig})
    plt.close(fig)
