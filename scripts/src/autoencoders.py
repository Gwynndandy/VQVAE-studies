import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label





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
history_vqvae = {
    "train_loss": [],
    "val_loss": [],
    "train_recon": [],
    "val_recon": [],
    "train_vq": [],
    "val_vq": [],
    "lr": [],
}
read_yaml_vqvae():
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
    return args

get_model(config)
    model = model(args)
    optsched = asdflakjds

    return model,op,sched

class autoencoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size = 5, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size = 9, stride=5, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(64, 32, kernel_size = 9, stride=5, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size = 5, stride=4, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(1, 1, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.Upsample(size=200, mode='linear', align_corners=False)
        )
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)
history_autoencoder = {
    "train_loss": [],
    "val_loss": [],
    "lr": [],
}
read_yaml_autoencoder():
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
    return args

get_autoencoder()
    model = model(asdfkjh)

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
    return mdoelstuff, kwargs