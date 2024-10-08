import os
import torch
from classification import Exp_Classification
from exp_anomaly_detection import Exp_Anomaly_Detection
import random
import numpy as np
from config import CONFIG

if __name__ == "__main__":
    seed = CONFIG['seed']
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    setting = "{}_{}_{}_{}_bs{}_{}hz_{}_{}_sl{}_lr{}_pl{}_dm{}_nh{}_el{}_df{}_fc{}_eb{}_{}_{}_seed{}_{}".format(
                CONFIG['task_name'],
                CONFIG['model_id'],
                CONFIG['model'],
                CONFIG['data'],
                CONFIG['batch_size'],
                CONFIG['signal_hz'],
                CONFIG['monitor'],
                CONFIG['loss'],
                CONFIG['seq_len'],
                CONFIG['learning_rate'],
                CONFIG['pred_len'],
                CONFIG['d_model'],
                CONFIG['n_heads'],
                CONFIG['e_layers'],
                CONFIG['d_ff'],
                CONFIG['factor'],
                CONFIG['embed'],
                CONFIG['activation'],
                CONFIG['des'],
                CONFIG['seed'],
                CONFIG['augmentations'],
            )
    

    if CONFIG['task_name'] == "anomaly_detection":
        exp = Exp_Anomaly_Detection(CONFIG)
         
    elif CONFIG['task_name'] == "classification":
        exp = Exp_Classification(CONFIG)
    
    if CONFIG['is_training']:
        
        print("==============================   Start Training   ==============================")
        exp.train(setting)
    
        print("==============================      Testing      ==============================")
        exp.test(setting)
        
        torch.cuda.empty_cache()
    
    else:
        print("==============================      Testing      ==============================")
        exp.test(setting, test=1)
        
        torch.cuda.empty_cache()
