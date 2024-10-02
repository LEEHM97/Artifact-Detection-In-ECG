import os
import torch
# from classification import Exp_Classification
from Ensemble_basic import BasicModel
from Ensemble_classification import EnsembleModel
import random
import numpy as np
from config import CONFIG

if __name__ == "__main__":
    for ii in range(CONFIG['itr']):

        # seed = CONFIG['seed']
        seed = 41 + ii
        
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
        
        exp1 = BasicModel(CONFIG, CONFIG['path1'])
        exp2 = BasicModel(CONFIG, CONFIG['path2'])
        exp3 = BasicModel(CONFIG, CONFIG['path3'])
        exp4 = BasicModel(CONFIG, CONFIG['path4'])
        exp5 = BasicModel(CONFIG, CONFIG['path5'])
        
        models = [exp1.swa_model, exp2.swa_model, exp3.swa_model, exp4.swa_model, exp5.swa_model]
        
        exp = EnsembleModel(CONFIG, models)
        
        if CONFIG['is_training']:
            
            # print("\n==============================   Start Training   ==============================")
            # exp.train(setting)
        
            print("===============================      Testing      ==============================")
            exp.test(setting)
            
            torch.cuda.empty_cache()
        
        else:
            print("===============================      Testing      ==============================")
            exp.test(setting, test=1)
            
            torch.cuda.empty_cache()