import os
import torch
import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class KMediconLoader(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X = self.X.reshape(-1,2500,12)
        self.X = normalize_batch_ts(self.X)

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index])
        y = torch.from_numpy(np.asarray(self.y[index]))
        y = torch.nn.functional.one_hot(y.reshape(-1,).to(torch.long), num_classes=2)
        
        return X, y

    def __len__(self):
        return len(self.y)
    

class PublicTest(Dataset):
    def __init__(self, data_path):
        with h5py.File(data_path, 'r') as f:
            ecg = f['ecg'][:]
            label = f['label'][:]
        
        self.X = ecg
        self.y = label
        self.X = self.X.reshape(-1,2500,12)
        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=250, lowcut=0.5, highcut=45)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.X)
    
    
def normalize_ts(ts):
    """normalize a time-series data

    Args:
        ts (numpy.ndarray): The input time-series in shape (timestamps, feature).

    Returns:
        ts (numpy.ndarray): The processed time-series.
    """
    # scaler = StandardScaler()
    # scaler.fit(ts)
    # ts = scaler.transform(ts)

    # Min-Max
    for i in range(12):
        signal = ts[:, i]
        ts[:, i] = (signal - signal.mean()) / signal.std()
    
    return ts    
    
    
def normalize_batch_ts(batch):
    """normalize a batch of time-series data

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (n_samples, timestamps, feature).

    Returns:
        A batch of processed time-series.
    """
    return np.array(
        list(map(normalize_ts, batch))
    )