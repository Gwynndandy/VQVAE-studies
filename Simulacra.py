import numpy as np
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

