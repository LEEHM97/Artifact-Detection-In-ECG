import os
import torch
import numpy as np
from scipy import interpolate
import pickle
import h5py

from torch.utils.data import Dataset
from sklearn.utils import shuffle
from config import CONFIG

    
class KMediconLoader(Dataset):
    def __init__(self, root_path, flag=None):
        
        self.ecg, self.label = get_data_from_pkl(root_path, flag='train')
        
        # list of IDs for training, val sets
        self.train_ids, self.val_ids = self.load_train_val_test_list(self.label, CONFIG['split_ratio'])

        self.X, self.y = self.load_data(self.ecg, self.label, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label, ratio):
        norm_list = list(
            list(np.where(label[:] == 0)[0])
        )  # Normal ECG IDs
        arti_list = list(
            list(np.where(label[:] == 1)[0])
        )  # Artifact ECG IDs

        train_ids = (
            norm_list[: int(ratio * len(norm_list))]
            + arti_list[: int(ratio * len(arti_list))]
        )
        val_ids = (
            norm_list[int(ratio * len(norm_list)) : ]
            + arti_list[int(ratio * len(arti_list)) : ]
        )

        return train_ids, val_ids

    def load_data(self, ecg, label, flag=None):
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
                    
        X = ecg[ids]        
        y = label[ids]
        X, y = shuffle(X, y, random_state=42)

        return X, y

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index])
        y = torch.from_numpy(np.asarray(self.y[index]))
        y = torch.nn.functional.one_hot(y.reshape(-1,).to(torch.long), num_classes=2)
        
        return X, y

    def __len__(self):
        return len(self.y)
    
    
class KMediconPrivateLoader(Dataset):
    def __init__(self, data_path):
        self.X = get_data_from_pkl(data_path, flag='test')
        
        # pre_process
        self.X = normalize_batch_ts(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index])

    def __len__(self):
        return len(self.X)  
    
    
def normalize_ts(ts):
    """normalize a time-series data

    Args:
        ts (numpy.ndarray): The input time-series in shape (timestamps, feature).

    Returns:
        ts (numpy.ndarray): The processed time-series.
    """
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
    

# resampling to 250Hz
def resampling(array, freq, kind='linear'):
    t = np.linspace(1, len(array), len(array))
    f = interpolate.interp1d(t, array, kind=kind)
    t_new = np.linspace(1, len(array), int(len(array)/freq * 250))
    new_array = f(t_new)
    
    return new_array


def get_data_from_pkl(root_path, flag):
    if flag=='train':
        with open(os.path.join(root_path, "Signal_Train.pkl"), 'rb') as f:
            ecg = pickle.load(f)
                
        with open(os.path.join(root_path, "Target_Train.pkl"), 'rb') as f:
            label = pickle.load(f)
    
        p_label = label.Target.values
    
    elif flag=='test':
        with open(os.path.join(root_path, "Signal_Test_Private.pkl"), 'rb') as f:
            ecg = pickle.load(f)
    
    ecg = np.array(ecg)
    
    p_ecg = []
    for ecg_data in ecg:
        sub = []
        trial = []
        for ch in range(ecg_data.shape[1]):
            data = resampling(ecg_data[:,ch], freq=500, kind='linear')
            trial.append(data)
            
        trial = np.array(trial).T
        sub.append(trial)

        sub = np.array(sub)
        sub = sub.reshape(-1, 2500, sub.shape[-1])

        p_ecg.append(sub)
        
    p_ecg = np.array(p_ecg).squeeze(1)

    if flag=='train':    
        return p_ecg, p_label
    
    else:
        return p_ecg
