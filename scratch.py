import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scripts.src.data_handling as data

x_train_data_path = r"/home/maniacalm/bin/VQVAE_studies/C:\Users\vanth\Desktop\Workspace\AUTOENCODER\processed_data_LATEST\x_train_U.npy"
y_train_data_path = r"/home/maniacalm/bin/VQVAE_studies/C:\Users\vanth\Desktop\Workspace\AUTOENCODER\processed_data_LATEST\y_train_AE_U.npy"
x_test_data_path = r"/home/maniacalm/bin/VQVAE_studies/C:\Users\vanth\Desktop\Workspace\AUTOENCODER\processed_data_LATEST\x_test_U.npy"
y_test_data_path = r"/home/maniacalm/bin/VQVAE_studies/C:\Users\vanth\Desktop\Workspace\AUTOENCODER\processed_data_LATEST\y_test_AE_U.npy"
train_loader,val_loader,test_loader = data.npy_ds(x_train_data_path,y_train_data_path,x_test_data_path,y_test_data_path)
batch = next(iter(test_loader))
print(batch["x"].shape)
x = batch["x"]
y = batch["y"]
plt.plot(x[0])
plt.plot(y[0])
plt.savefig(f"output.png", dpi=150, bbox_inches="tight")
plt.clf()
print(y[0])
# x_train=np.load(x_train_data_path,allow_pickle=True)
# y_train=np.load(y_train_data_path,allow_pickle=True)
# for i in [0,1,2,-1,-2,-3]:
    # plt.plot(x_train[i])
    # plt.plot(y_train[i])
    # plt.savefig(f"output{i}.png", dpi=150, bbox_inches="tight")
    # plt.clf()