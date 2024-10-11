import os
import torch
import random
import h5py
import numpy as np
from classification import Exp_Classification
from config import CONFIG
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    # Get Dataset
    with h5py.File(os.path.join(CONFIG['root_path'], "processed_features.h5"), 'r') as f:
        ecg = f['ecg'][:]
        label = f['label'][:]
    
    
    for ii in range(CONFIG['itr']):
        # seed = CONFIG['seed']
        seed = 41+ii
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        CONFIG['seed'] = seed
        setting = "{}_{}_{}_{}_bs{}_sl{}_lr{}_pl{}_dm{}_nh{}_el{}_df{}_fc{}_eb{}_{}_seed{}".format(
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
                )
        
        exp = Exp_Classification(CONFIG)
        
        if CONFIG['is_training']:
            print(setting)
            print("==============================   Start Training   ==============================")
            exp.train(setting)
        
            print("===============================      Testing      ==============================")
            exp.test(setting)
            
            torch.cuda.empty_cache()
        
        else:
            print("===============================      Testing      ==============================")
            exp.test(setting, test=1)
            
            torch.cuda.empty_cache()