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
target_snr, pulse_length = float(sys.argv[1]),200
training_length,test_length,val_length = 10000,1000,1000
training_seed,test_seed,val_seed = 0,training_length*2,training_length*3
batch_size = 64

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
    project="ROI Init Testing",
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

train_ds = sc.simulacra_dataset(target_snr, training_length, training_seed, pulse_length,roi=True)
val_ds = sc.simulacra_dataset(target_snr, val_length, val_seed, pulse_length,roi=True)
test_ds = sc.simulacra_dataset(target_snr, test_length, test_seed, pulse_length,roi=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

batch = next(iter(train_loader))
batch["x"].shape, batch["y"].shape

time_periods = 200
time_periods = 200
class disciminator(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.1),

            nn.Conv1d(32, 64, kernel_size=9, stride=1), 
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=0.2),

            nn.Flatten(), 
            )
        self.fc = nn.Linear(64, 1)      

    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return torch.sigmoid(x)

model = disciminator().to(device)

print(model)
batch = next(iter(train_loader))

x = batch["x"].to(device)
y = batch["y"].to(device)

with torch.no_grad():
    y_hat= model(x)

print("x:", x.shape)
print("y:", y.shape)
print("y_hat:", y_hat.shape)

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

def run_epoch(loader, train=True, max_grad_norm=1.0):
    model.train() if train else model.eval()

    total_loss = 0.0
    skipped = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train):
            y_hat = model(x)

            loss = F.smooth_l1_loss(y_hat, y)

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

    n = max(1, len(loader) - skipped)

    return {
        "loss": total_loss / n,
        "skipped": skipped,
    }

best_val = float("inf")
best_state = None

history = {
    "train_loss": [],
    "val_loss": [],
    "lr": [],
}

for epoch in range(1, num_epochs + 1):

    train_stats = run_epoch(
        train_loader,
        train=True,
        max_grad_norm=1.0,
    )

    val_stats = run_epoch(
        val_loader,
        train=False,
        max_grad_norm=1.0,
    )

    scheduler.step(val_stats["loss"])

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_stats["loss"])
    history["val_loss"].append(val_stats["loss"])
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
        f"train {train_stats['loss']:.6f} | "
        f"val {val_stats['loss']:.6f} | "
        f"skipped {train_stats['skipped']}"
    )

    if bad_epochs >= patience:
        print("Early stopping.")
        break

model.load_state_dict(best_state)
model.to(device)

test_stats = run_epoch(test_loader, train=False)
print(test_stats)

model.eval()

batch = next(iter(test_loader))

x = batch["x"].to(device)
y = batch["y"].to(device)

with torch.no_grad():
    y_hat = model(x)
