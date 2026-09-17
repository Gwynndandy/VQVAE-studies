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


wandb.login()

# Project that the run is recorded to
project = "my-awesome-project"

# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    # Training code here
    # Log values to W&B with run.log()
    run.log({"accuracy": 0.9, "loss": 0.1})

import random

wandb.login()

# Project that the run is recorded to
project = "my-awesome-project"

# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    offset = random.random() / 5
    print(f"lr: {config['lr']}")

    # Simulate a training run
    for epoch in range(2, config['epochs']):
        acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
        loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
        print(f"epoch={config['epochs']}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})


train_ds = sc.simulacra_dataset(target_snr, training_length, training_seed, pulse_length)
val_ds = sc.simulacra_dataset(target_snr, val_length, val_seed, pulse_length)
test_ds = sc.simulacra_dataset(target_snr, test_length, test_seed, pulse_length)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

batch = next(iter(train_loader))
batch["x"].shape, batch["y"].shape