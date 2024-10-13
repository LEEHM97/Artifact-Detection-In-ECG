import os
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader import KMediconPrivateLoader
from config import CONFIG
from classification import Exp_Classification


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
    
    testset = KMediconPrivateLoader(CONFIG['root_path'])
    test_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        )
    
    model_path = CONFIG['ckpt_path']
    
    exp.swa_model.load_state_dict(torch.load(model_path))
    exp.swa_model.eval()

    device = torch.device("cuda:0")

    total_loss = []
    preds = []

    with torch.no_grad():
        for i, (batch_x) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.float().to(device)

            outputs = exp.swa_model(batch_x, None, None, None)

            pred = outputs.detach().cpu()

            preds.append(outputs.detach())

    preds = torch.cat(preds, 0)

    probs = torch.nn.functional.sigmoid(preds)    
    probs = probs.cpu().numpy()

    df = pd.DataFrame({"possibility":probs[:, 1]})

    save_xlsx = pd.ExcelWriter("./RTFACT_241013_Private_Possibility.xlsx")
    df.to_excel(save_xlsx, index = False) 
    
    save_xlsx.save()