import uuid
import torch
import torch.nn.functional as F
import sys
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def init(history,run_name,path):
    run_id = run_name + "-" + str(uuid.uuid4().hex)[:6]

    filepath = f"{path}/{run_id}/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    csv_filename = f"{filepath}/run_data.csv"

    file = open(csv_filename, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(["epoch", "current_lr", "lambda_vq", "train_loss",'val_loss','val_recon','val_vq','skipped'])
    file.close()
    return csv_filename,filepath

def vae_run_epoch(model, loader, optimizer, history, train=True, lambda_vq=0.25, max_grad_norm=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train() if train else model.eval()
    label = "train" if train else "val"

    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    skipped = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train):
            y_hat = model(x)

            recon_loss = F.huber_loss(y_hat, y, delta=0.5)
            loss =  recon_loss

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

    n = max(1, len(loader) - skipped)
    history[f"{label}_loss"].append(total_loss/n)
    history["skipped"].append(skipped)
    return {
        "loss": total_loss / n,
        "skipped": skipped,
    }
   
def vae_train(model,scheduler,optimizer,history,path,num_epochs,train_loader,val_loader,csv_filename):
    file = open(csv_filename, mode='a', newline='')
    writer = csv.writer(file)
    best_val = float("inf")
    best_state = None
    patience = 15
    bad_epochs = 5
    for epoch in range(1, num_epochs + 1):


        current_lr = optimizer.param_groups[0]["lr"]

        train_stats = vae_run_epoch(
            model,
            train_loader,
            optimizer,
            history,
            train=True,
            max_grad_norm=1.0,
        )

        val_stats = vae_run_epoch(
            model,
            val_loader,
            optimizer,
            history,
            train=False,
            max_grad_norm=1.0,
        )

        scheduler.step(val_stats["loss"])
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
        
        writer.writerow(history.values())

        if bad_epochs >= patience:
            print("Early stopping.")
            break
    file.close()

def vae_eval(model,optimizer,path,test,history):
    test_stats = vae_run_epoch(model,test,optimizer,history, train=False, lambda_vq=0.25)

    with open(f"{path}/test_stats.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(test_stats.keys())  # Write header
        writer.writerow(test_stats.values())  # Write values


    model.eval()
    batch = next(iter(test))

    x = batch["x"]
    y = batch["y"]

    with torch.no_grad():
        y_hat = model(x)

    for idx in range(min(5,len(x))):
        noisy = x[idx, 0].cpu().numpy()
        clean = y[idx, 0].cpu().numpy()
        pred = y_hat[idx, 0].cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(clean, label="clean target", linewidth=2)
        ax.plot(noisy, label="noisy input", alpha=0.6)
        ax.plot(pred, label="VAE output", linewidth=2)

        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        plt.savefig(f"{path}/output_{idx}.png")
        plt.close(fig)

    torch.save(model.state_dict(), f"{path}/model_dict.pth")
    torch.save(model, f"{path}/model.pth")

def vq_run_epoch(model, optimizer, loader, history, train=True, lambda_vq=0.25, max_grad_norm=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train() if train else model.eval()
    label = "train" if train else "val"

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

    history[f"{label}_loss"].append(total_loss/n)
    history[f"{label}_recon"].append(total_recon/n)
    history[f"{label}_vq"].append(total_vq / n)
    history["skipped"].append(skipped)
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "vq": total_vq / n,
        "skipped": skipped,
    }

def vq_train(model,scheduler,optimizer,history,path,num_epochs,train_loader,val_loader,csv_filename):
    file = open(csv_filename, mode='a', newline='')
    writer = csv.writer(file)
    best_val = float("inf")
    best_state = None
    patience = 15
    bad_epochs = 5

    for epoch in range(1, num_epochs + 1):

        # Warm up VQ loss so it does not dominate early training
        lambda_vq = min(0.15, 0.15 * epoch / 10)

        train_stats = vq_run_epoch(
            model,
            optimizer,
            train_loader,
            history,
            train=True,
            lambda_vq=lambda_vq,
            max_grad_norm=1.0,
        )

        val_stats = vq_run_epoch(
            model,
            optimizer,
            val_loader,
            history,
            train=False,
            lambda_vq=lambda_vq,
            max_grad_norm=1.0,
        )

        scheduler.step(history["val_loss"][-1])

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)


        if history["val_loss"][-1] < best_val:
            best_val = history["val_loss"][-1]
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
            f"train {history['train_loss'][-1]:.6f} | "
            f"val {history['val_loss'][-1]:.6f} | "
            f"recon {history['val_recon'][-1]:.6f} | "
            f"vq {history['val_vq'][-1]:.6f} | "
            f"skipped {history['skipped'][-1]}"
        )
        
        writer.writerow([epoch, current_lr, lambda_vq, history['train_loss'][-1],history['val_loss'][-1],history['val_recon'][-1],history['val_vq'][-1],history['skipped'][-1]])

        if bad_epochs >= patience:
            print("Early stopping.")
            break
    model.load_state_dict(best_state)
    file.close()

def vq_eval(model,optimizer,path,test,history):
    test_stats = vq_run_epoch(model,optimizer,test,history, train=False, lambda_vq=0.25)

    with open(f"{path}/test_stats.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(test_stats.keys())  # Write header
        writer.writerow(test_stats.values())  # Write values


    model.eval()
    batch = next(iter(test))

    x = batch["x"]
    y = batch["y"]

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

        plt.savefig(f"{path}/output_{idx}.png")
        plt.close(fig)

    torch.save(model.state_dict(), f"{path}/model_dict.pth")
    torch.save(model, f"{path}/model.pth")

def cnn_run_epoch(model, loader, optimizer, history, train=True, lambda_vq=0.25, max_grad_norm=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train() if train else model.eval()
    label = "train" if train else "val"

    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    skipped = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train):
            y_hat = model(x)

            recon_loss = F.cross_entropy(y_hat, y, delta=0.5)
            loss =  recon_loss

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
    history[f"{label}_loss"].append(total_loss/n)
    history["skipped"].append(skipped)
    return {
        "loss": total_loss / n,
        "skipped": skipped,
    }
   
def cnn_train(model,scheduler,optimizer,history,path,num_epochs,train_loader,val_loader,csv_filename):
    file = open(csv_filename, mode='a', newline='')
    writer = csv.writer(file)
    best_val = float("inf")
    best_state = None
    patience = 15
    bad_epochs = 5
    for epoch in range(1, num_epochs + 1):


        current_lr = optimizer.param_groups[0]["lr"]

        train_stats = vae_run_epoch(
            model,
            train_loader,
            optimizer,
            history,
            train=True,
            max_grad_norm=1.0,
        )

        val_stats = vae_run_epoch(
            model,
            val_loader,
            optimizer,
            history,
            train=False,
            max_grad_norm=1.0,
        )

        scheduler.step(val_stats["loss"])
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
        
        writer.writerow(history.values())

        if bad_epochs >= patience:
            print("Early stopping.")
            break
    file.close()

def cnn_eval(model,optimizer,path,test,history):
    test_stats = vae_run_epoch(model,test,optimizer,history, train=False, lambda_vq=0.25)

    with open(f"{path}/test_stats.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(test_stats.keys())  # Write header
        writer.writerow(test_stats.values())  # Write values


    model.eval()
    batch = next(iter(test))

    x = batch["x"]
    y = batch["y"]

    with torch.no_grad():
        y_hat = model(x)

    
    print(torch.argmax(y_hat, dim=1),torch.argmax(y, dim=1))

    cm = confusion_matrix(torch.argmax(y_hat, dim=1),torch.argmax(y, dim=1))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{path}/confusion_matrix.png")
    plt.close(fig)

    torch.save(model.state_dict(), f"{path}/model_dict.pth")
    torch.save(model, f"{path}/model.pth")

