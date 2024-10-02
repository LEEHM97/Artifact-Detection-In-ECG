from data_factory import data_provider
from exp_basic import Exp_Basic
import torch
import torch.nn as nn
from torch import optim
import warnings
import numpy as np
import random
from config import CONFIG

warnings.filterwarnings("ignore")


class BasicModel(Exp_Basic):
    def __init__(self, args, ckpt_path):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args['swa']
        self.ckpt_path = ckpt_path
        
        try:
            print(f'Loading "{ckpt_path}"')
            self.swa_model.load_state_dict(torch.load(ckpt_path))
        except:
            print("Failed to load checkpoint")
            
    def _build_model(self):
        # model input depends on data
        test_data, test_loader = self._get_data(flag="TEST")
        self.args['seq_len'] = test_data.max_seq_len  # redefine seq_len
        self.args['pred_len'] = 0
        self.args['enc_in'] = test_data.X.shape[2]  # redefine enc_in
        self.args['num_class'] = len(np.unique(test_data.y))
        
        # model init
        model = (
            self.model_dict[self.args['model']].Model(self.args).float()
        )  # pass args to model
        
        if self.args['use_multi_gpu'] and self.args['use_gpu']:
            model = nn.DataParallel(model, device_ids=self.args['device_ids'])
            
        return model
    
    def _get_data(self, flag):
        random.seed(self.args['seed'])
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader