import os
import torch
import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class KMediconLoader(Dataset):
    def __init__(self, root_path, flag=None):
        # features.h5 : 500 Hz Original sampling rate file
        # processed_features.h5 : 250 Hz Resampled sampling rate file
        # data_file = os.path.join(root_path, "features.h5")
        data_file = os.path.join(root_path, "processed_features.h5")
        with h5py.File(data_file, 'r') as f:
            ecg = f['ecg'][:]
            label = f['label'][:]        
            ### 🖌️ Sample test
            # ecg = ecg[15:45]
            # label = label[15:45]

        self.ecg = ecg
        self.label = label
        self.root_path = root_path

        a, b = 0.6, 0.8
        # a, b = 0.8, -1

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label, a, b
        )

        self.X, self.y = self.load_ecgh5(self.ecg, self.label, flag=flag)

        self.X = self.X.reshape(-1, 2500, 12)
        # self.X = self.X.reshape(-1, 5000, 12)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=250, lowcut=0.5, highcut=45)
        # print(f'self.X.shape: {self.X.shape}')

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        norm_list = list(
            list(np.where(label[:] == 0)[0])
        )  # Normal ECG IDs
        arti_list = list(
            list(np.where(label[:] == 1)[0])
        )  # Artifact ECG IDs

        train_ids = (
            norm_list[: int(a * len(norm_list))]
            + arti_list[: int(a * len(arti_list))]
        )
        val_ids = (
            norm_list[int(a * len(norm_list)) : int(b * len(norm_list))]
            + arti_list[int(a * len(arti_list)) : int(b * len(arti_list))]
            # norm_list[int(a * len(norm_list)) : b]
            # + arti_list[int(a * len(arti_list)) : b]
        )
        test_ids = (
            norm_list[int(b * len(norm_list)) :]
            + arti_list[int(b * len(arti_list)) :]
            # norm_list[b:]
            # + arti_list[b:]
        )

        return train_ids, val_ids, test_ids

    def load_ecgh5(self, ecg, label, flag=None):
        """
        Loads 12-lead ECG data from h5 files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data & label files
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        # filenames = []

        # The first column is the label; the second column is the patient ID
        # subject_label = np.load(label_path)
        # for filename in os.listdir(data_path):
        #     filenames.append(filename)
        # filenames.sort()

        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        # else:
            # ids = subject_label[:, 1]
            # print("all ids:", ids)

        # for j in range(len(filenames)):
        #     trial_label = subject_label[j]
        #     path = data_path + filenames[j]
        #     subject_feature = np.load(path)
        #     for trial_feature in subject_feature:
        #         # load data by ids
        #         if j in ids:  # id starts from 1, not 0.
        #             feature_list.append(trial_feature)
        #             label_list.append(trial_label)
                    
        # reshape and shuffle
        X = ecg[ids]        
        y = label[ids]
        X, y = shuffle(X, y, random_state=42)

        return X, y  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)
    

class PublicTest(Dataset):
    def __init__(self, data_path):
        with h5py.File(data_path, 'r') as f:
            ecg = f['ecg'][:]
            label = f['label'][:]
        
        self.X = ecg
        self.y = label
        self.X = self.X.reshape(-1, 2500, 12)
        # self.X = self.X.reshape(-1, 5000, 12)
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
    scaler = StandardScaler()
    scaler.fit(ts)
    ts = scaler.transform(ts)
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