mport numpy as np
import numpy.random as rng
from torch.utils.data import Dataset
import torch

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
train_ds = sc.simulacra_dataset(target_snr, training_length, training_seed, pulse_length)
val_ds = sc.simulacra_dataset(target_snr, val_length, val_seed, pulse_length)
test_ds = sc.simulacra_dataset(target_snr, test_length, test_seed, pulse_length)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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

        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y).float(),
        }
clean_paths = [x.name for x in os.scandir(data_path) if check_clnsig(x.name) is not None]
signal_paths = [x.name for x in os.scandir(data_path) if check_signal(x.name) is not None]
train_noise = []
train_signal = []
for cln,sig in zip(clean_paths,signal_paths):
    a=np.load(sig,allow_pickle=True)
    c=np.load(cln,allow_pickle=True)
    nwf=a.size
    ntcks=len(a[0])-54
    for j in range(nwf-50):
        ya=[]
        yc=[]
        j=0
        for i in range(0,ntcks):
            lab='tck_'+str(i)
            ya.append(a[j][lab])
            yc.append(c[j][lab])
        train_noise.append(ya)
        train_signal.append(yc)
    test_noise = []
    test_signal = []
    for j in range(nwf-50,nwf-1):
        ya=[]
        yc=[]
        j=0
        for i in range(0,ntcks):
            lab='tck_'+str(i)
            ya.append(a[j][lab])
            yc.append(c[j][lab])
    test_noise.append(ya)
    test_signal.append(yc)
train_noise_c = np.reshape(np.array(train_noise), (-1, 1,512))
train_signal_c = np.reshape(np.array(train_signal), (-1, 1,512))
test_noise_c = np.reshape(np.array(test_noise), (-1, 1,512))
test_signal_c = np.reshape(np.array(test_signal), (-1, 1,512))
train_ds_full = NoisyToClean1DDataset(
    noisy=train_noise_c,
    clean=train_signal_c,
    normalize=True,
)
test_ds = NoisyToClean1DDataset(
    noisy=test_noise_c,
    clean=test_signal_c,
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


