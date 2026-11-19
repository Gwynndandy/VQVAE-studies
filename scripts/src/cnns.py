import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

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

def read_yaml_roi(yaml_path):
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

def get_roi(config):
    model = disciminator()

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