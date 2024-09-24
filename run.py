import os
import torch
from classification import Exp_Classification
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
    
    setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}".format(
                CONFIG['task_name'],
                CONFIG['model_id'],
                CONFIG['model'],
                CONFIG['data'],
                CONFIG['features'],
                CONFIG['seq_len'],
                CONFIG['label_len'],
                CONFIG['pred_len'],
                CONFIG['d_model'],
                CONFIG['n_heads'],
                CONFIG['e_layers'],
                CONFIG['d_layers'],
                CONFIG['d_ff'],
                CONFIG['factor'],
                CONFIG['embed'],
                CONFIG['distil'],
                CONFIG['des'],
                CONFIG['seed'],
            )
    
    exp = Exp_Classification(CONFIG)
    
    if CONFIG['is_training']:
        
        print("==============================   Start Training   ==============================")
        exp.train(setting)
    
        print("===============================      Testing      ==============================")
        exp.test(setting)
        
        torch.cuda.empty_cache()
    
    else:
        print("===============================      Testing      ==============================")
        exp.test(setting, test=1)
        
        torch.cuda.empty_cache()