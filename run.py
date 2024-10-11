import os
import torch
import random
import h5py
import numpy as np
from classification import Exp_Classification
from config import CONFIG
from sklearn.model_selection import StratifiedKFold
from data_loader import KMediconLoader, PublicTest
from data_factory import collate_fn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Get train dataset
    with h5py.File(os.path.join(CONFIG['root_path'], "processed_features.h5"), 'r') as f:
        ecg = f['ecg'][:]
        label = f['label'][:]
        
    # Get test dataset
    with h5py.File("./dataset/KMedicon/public_test2.h5", 'r') as f:
        test_X = f['ecg'][:]
        test_y = f['label'][:]
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for idx, (train_index, valid_index) in enumerate(kf.split(ecg, label)):
        if idx == 0:
            continue

        seed = CONFIG['seed']
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        CONFIG['seed'] = seed
        setting = "{}_{}_{}_{}_bs{}_sl{}_lr{}_pl{}_dm{}_nh{}_el{}_df{}_fc{}_eb{}_{}_seed{}_fold{}".format(
                    CONFIG['task_name'],
                    CONFIG['model_id'],
                    CONFIG['model'],
                    CONFIG['data'],
                    CONFIG['batch_size'],
                    CONFIG['seq_len'],
                    CONFIG['learning_rate'],
                    CONFIG['pred_len'],
                    CONFIG['d_model'],
                    CONFIG['n_heads'],
                    CONFIG['e_layers'],
                    CONFIG['d_ff'],
                    CONFIG['factor'],
                    CONFIG['embed'],
                    CONFIG['des'],
                    CONFIG['seed'],
                    idx+1
                )
        
        
        train_X = ecg[train_index]
        train_y = label[train_index]
        val_X = ecg[valid_index]
        val_y = label[valid_index]
        
        train_data = KMediconLoader(train_X, train_y)
        vali_data = KMediconLoader(val_X, val_y)
        test_data = PublicTest(test_X, test_y)
        
        train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, 
                                  drop_last=False, collate_fn=lambda x: collate_fn(x, max_len=CONFIG['seq_len']))
        
        vali_loader = DataLoader(vali_data, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, 
                                  drop_last=False, collate_fn=lambda x: collate_fn(x, max_len=CONFIG['seq_len']))
        
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        
        exp = Exp_Classification(CONFIG, train_data, train_loader, vali_data, vali_loader, test_data, test_loader)
        
        if CONFIG['is_training']:
            print(setting)
            print("==============================   Start Training   ==============================")
            exp.train(setting)
        
            print("===============================      Testing      ==============================")
            exp.test(setting, test=1)
            
            torch.cuda.empty_cache()
        
        else:
            print("===============================      Testing      ==============================")
            exp.test(setting, test=1)
            
            torch.cuda.empty_cache()