def init(history,run_name,project,path)
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
    #run_name = f"SNR:{target_snr}-Run:{run.id}"
    run.name = run_name

    filepath = f"{sys.argv[2]}/{run_name}/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    csv_filename = f"{filepath}/run_data.csv"
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

train(model,history,path)
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

def eval(model,path,test_ds)
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