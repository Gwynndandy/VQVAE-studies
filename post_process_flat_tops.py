import os
import sys
import re
import pathlib
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import numpy as np
import copy
import torch.nn.functional as F

from VQ_VAE import VQVAE1DDenoiser 
import torch
from torch.utils.data import Dataset, DataLoader
import Simulacra as sc

from pylatex import (
    Alignat,
    Axis,
    Document,
    Figure,
    Math,
    Matrix,
    Plot,
    Section,
    Subsection,
    Tabular,
    TikZ,
    SubFigure,
    NoEscape
)
from pylatex.utils import italic


def import_yaml(yaml_path):
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
def add_current_plot(path,name,caption):
    name = name.replace(" ", "_")
    name = name.replace(".", ",")
    image_filename = os.path.join(path,"images",f"{name}.png")
    print(image_filename)
    plt.savefig(image_filename)
    with doc.create(Figure(position="h!")) as plot:
        plot.add_image(image_filename, width="120px")
        plot.add_caption(caption)
    plt.clf()
def get_snr(run_path):
    snr_regex = re.compile(r'/SNR:[-+]?([0-9]*\.[0-9]+|[0-9]+)')
    matches = snr_regex.search(run_path)
    if matches is not None:
        return float(matches.groups()[0])
    else:
        return None
def get_id(run_path):
    id_regex = re.compile(r'Run:([\s\S]*)')
    return str(id_regex.search(run_path).groups()[0])
def get_stats(run_path):
    with open(f"{run_path}/test_stats.csv") as csv_file: 
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        results = next(csv_reader)
    stats = {}
    for i in range(len(header)):
        stats[header[i]] = results[i]
    return stats
def get_data(target_snr):
    snr = float(target_snr)
    test_ds = sc.simulacra_dataset(snr, 32, 10000000, 512)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    batch = next(iter(test_loader))
    x = batch["x"]
    y = batch["y"]
    return x,y,x[0, 0].cpu().numpy(), y[0, 0].cpu().numpy()


if __name__ == "__main__":
    try:
        os.chdir(sys.argv[1])
    except FileNotFoundError:
        print("File Path Not Found, Check Path")

    geometry_options = {"tmargin": "5cm", "lmargin": "5cm"}
    doc = Document(geometry_options=geometry_options)

    path = pathlib.Path().resolve()
    os.makedirs(os.path.dirname(f"{path}/images/"), exist_ok=True)
    runs = [x[0] for x in os.walk(path) if get_snr(x[0]) is not None]
    runs = sorted(runs,key=get_snr)
    snrs = [get_snr(x) for x in runs]
    sorted_runs= {}

    for snr in sorted(set(snrs)):
        sorted_runs[snr] = [run for run in runs if get_snr(run) == snr]
    
    with doc.create(Section("Plot of Flattops:")):
        metrics = {}
        for snr in sorted(set(snrs)):
            n_code_temp = []
            flatness_temp = []
            loss_temp = []
            preds = []
            x,y,_,_ = get_data(snr)
            for run in sorted_runs[snr]:
                config = import_yaml(f"{run}/batch.yaml")
                in_ch,hid,z_ch,n_codes = config["in_ch"], config["hid"], config["z_ch"], config["n_codes"]
                model = VQVAE1DDenoiser(in_ch, hid, z_ch, n_codes)
                model.load_state_dict(torch.load(f"{run}/model_dict.pth"))

                with torch.no_grad():
                    y_hat, vq_loss, codes = model(x)
                pred = y_hat[0, 0].cpu().numpy()
                dy_dx = np.diff(pred)/len(pred)
                dy2_dx2 = np.diff(dy_dx)/len(dy_dx)
                integral = sum(dy2_dx2**2)
                recon_loss = F.huber_loss(y_hat, y, delta=0.5).item()

                n_code_temp.append(n_codes)
                flatness_temp.append(integral)
                loss_temp.append(recon_loss)
                preds.append(pred)
            metrics[snr] = (n_code_temp, flatness_temp, loss_temp, preds)

        fig, ax = plt.subplots(len(metrics.items()), figsize=(10, 10))
        j = 3
        for i,metric, in enumerate(metrics.items()):
            snr = metric[0]
            n_codes,flatnesses,_,preds = metric[1]
            [ax[i].plot(pred, label=f"{ncode},{flatness:.2e}", alpha=0.5) for ncode,flatness,pred in zip(n_codes[:j],flatnesses[:j],preds[:j])]
            ax[i].set_title(f"SNR: {snr}")
        [(axes.set_ylabel("loss"),axes.set_xlabel("n_codes"),axes.legend()) for axes in ax]
        add_current_plot(path,f"outputs",f"test")
        fig, ax = plt.subplots(figsize=(10, 10))
        for i,metric, in enumerate(metrics.items()):
            snr = metric[0]
            n_codes,flatnesses,loss,preds = metric[1]
            loss_normed = 50*(np.subtract(loss,min(loss)))/(max(loss)-min(loss))+10
            ax.scatter(n_codes, flatnesses, s=loss_normed, label=f"{snr}" )
        ax.set_ylabel("N_codes")
        ax.set_xlabel("Flattness")
        ax.legend()
        plt.legend()
        add_current_plot(path,f"n codes vs flattness scatter",f"test")
        fig, ax = plt.subplots(2, figsize=(10, 10))
        for i,metric, in enumerate(metrics.items()):
            snr = metric[0]
            n_codes,flatnesses,loss,preds = metric[1]
            ax[0].scatter(n_codes, np.divide(loss,np.abs(np.log(float(snr)))), label=f"{snr}")
            ax[1].scatter(n_codes, loss, label=f"{snr}")
        [(axes.set_ylabel("N_codes"),axes.set_xlabel("Loss"),axes.legend()) for axes in ax]
        ax[0].set_title("Scaled Loss (loss/snr)")
        ax[1].set_title("Unscalled Loss")
        plt.legend()
        add_current_plot(path,f"n codes vs loss",f"test")
        fig, ax = plt.subplots(2, figsize=(10, 10))
        for i,metric, in enumerate(metrics.items()):
            snr = metric[0]
            n_codes,flatnesses,loss,preds = metric[1]
            plot1 = ax[0].scatter(n_codes, np.multiply(flatnesses,float(snr)), label=f"{snr}")
            plot2 = ax[1].scatter(n_codes, flatnesses, label=f"{snr}")
        plt.legend()
        ax[0].set_title("Scaled Flattness (flattness/snr)")
        ax[1].set_title("Unscalled Flattness")
        add_current_plot(path,f"n codes vs flatness",f"test")
    doc.generate_pdf("full", clean_tex=False)