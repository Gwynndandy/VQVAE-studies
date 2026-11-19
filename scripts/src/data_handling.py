import numpy as np
import numpy.random as rng
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import glob

def get_gaussian(mu,sigma,pulse_length):
    x = np.arange(pulse_length)
    a = -0.5 * (x - mu)**2 / sigma**2
    return np.exp(a) / (sigma * np.sqrt(2 * np.pi))
class simulacra_dataset(Dataset):
    """simulacra of simulated data for DUNE low energy neutrino detection"""

    def __init__(self, target_snr, length, seed, pulse_length):
        """
        Arguments:
            target_snr (float): The Signal-to-Noise as ratio of root-mean-squared of both signal and noise
            length (int): Number of simulated samples in the dataset
            seed (int): The seed for the data
            pulse_length (int): The length of samples
        """
        self.target_snr = target_snr
        #self.sigma =(target_snr/5)/2
        self.sigma = target_snr
        self.length = length
        self.seed = seed
        self.pulse_length = pulse_length
        self.min_mu = int(pulse_length*0.1)
        self.max_mu = int(pulse_length*0.9)
        self.min_sigma = int(pulse_length*0.05)
        self.max_sigma = int(pulse_length*0.1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng.seed(idx+self.seed)
        target_snr = np.abs(self.target_snr +  rng.normal(loc=0.0, scale=self.sigma))
        num_pulse = rng.randint(1,3)
        bool_inverse = [rng.randint(0, 100) > 50 for _ in range(num_pulse)]

        clean = np.zeros([self.pulse_length])
        for inverse in bool_inverse:
            mu = rng.randint(self.min_mu,self.max_mu)
            sigma = rng.randint(self.min_sigma,self.max_sigma)
            clean += get_gaussian(mu,sigma,self.pulse_length)

            if inverse:
                weight = -rng.random()
                clean += get_gaussian(mu+sigma,sigma,self.pulse_length)*weight

        mu, sigma = 0, 0.32
        noise = np.random.normal(mu, sigma, self.pulse_length)
        
        signal_rms = np.sqrt(np.mean(clean**2))
        noise_rms = np.sqrt(np.mean(noise**2))
        
        weight = self.target_snr * noise_rms / signal_rms

        clean *= weight
        result = clean + noise
        normalization_constant =  max(max(result),max(clean))
        return {
            "x": torch.from_numpy(result/normalization_constant).float().unsqueeze(0),
            "y": torch.from_numpy(clean/normalization_constant).float().unsqueeze(0),
        }
def make_sc_ds(config):
    train_ds = simulacra_dataset(config["target_snr"], config["training_length"], config["training_seed"], config["pulse_length"])
    val_ds = simulacra_dataset(config["target_snr"], config["val_length"], config["val_seed"], config["pulse_length"])
    test_ds = simulacra_dataset(config["target_snr"], config["test_length"], config["test_seed"], config["pulse_length"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
    return train_loader,val_loader,test_loader

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

        if isinstance(y, (np.ndarray)):
            return {
                "x": torch.from_numpy(x).float().unsqueeze(0),
                "y": torch.from_numpy(y).float().unsqueeze(0),
            }
        else:
            return {
                "x": torch.from_numpy(x).float().unsqueeze(0),
                "y": torch.tensor(y).float().unsqueeze(0),
            }

def npy_ds(x_train_path,y_train_path,x_test_path,y_test_path):
    x_train=np.load(x_train_path,allow_pickle=True)
    y_train=np.load(y_train_path,allow_pickle=True)
    x_test=np.load(x_test_path,allow_pickle=True)
    y_test=np.load(y_test_path,allow_pickle=True)
    train_ds_full = NoisyToClean1DDataset(
        noisy=x_train,
        clean=y_train,
        normalize=True,
    )
    test_ds = NoisyToClean1DDataset(
        noisy=x_test,
        clean=y_test,
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
    return train_loader,val_loader,test_loader

def make_np_ds(config,type):
    dataset_path = config["dataset_path"]
    noise_train_path = glob.glob(os.path.join(dataset_path, "*x_train_U.npy"))[0]
    noise_test_path = glob.glob(os.path.join(dataset_path, "*x_test_U.npy"))[0]
    sig_train_path = glob.glob(os.path.join(dataset_path, "*y_train_" + type + "_U.npy"))[0]
    sig_test_path = glob.glob(os.path.join(dataset_path, "*y_test_" + type + "_U.npy"))[0]
    train_loader,val_loader,test_loader = npy_ds(noise_train_path,sig_train_path,noise_test_path,sig_test_path)
    return train_loader,val_loader,test_loader

