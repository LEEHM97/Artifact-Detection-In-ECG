from data_factory import data_provider
from exp_basic import Exp_Basic
from utils import EarlyStopping, mcc_score, CosineAnnealingWarmUpRestarts, FocalLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from config import CONFIG

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def train(self, model, setting):
    train_data, train_loader = self._get_data(flag="TRAIN")
    vali_data, vali_loader = self._get_data(flag="VAL")
    test_data, test_loader = self._get_data(flag="TEST")
    print(train_data.X.shape)
    print(train_data.y.shape)
    print(vali_data.X.shape)
    print(vali_data.y.shape)
    print(test_data.X.shape)
    print(test_data.y.shape)

    path = (
        "./checkpoints/"
        + setting
        + "/"
    )
    if not os.path.exists(path):
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(
        patience=self.args['patience'], verbose=True, delta=1e-5
    )

    model_optim = self._select_optimizer()
    criterion = self._select_criterion()
    scheduler = self._select_scheduler(model_optim)

    for epoch in range(self.args['train_epochs']):
        iter_count = 0
        train_loss = []

        self.models.train()
        epoch_time = time.time()
        
        print("[Train Step]")
        for i, (batch_x, label, padding_mask) in enumerate(tqdm(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)
            padding_mask = padding_mask.float().to(self.device)
            label = label.to(self.device)

            outputs = self.forward(batch_x, padding_mask, None, None)
            
            label_ = nn.functional.one_hot(label.long(), num_classes=2)
            loss = criterion(outputs, label_.float())
            train_loss.append(loss.item())

            loss.backward()
            # nn.utils.clip_grad_norm_(self.models.parameters(), max_norm=4.0)
            model_optim.step()

        # self.models.update_parameters(self.models)

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        
        print("\n[Validation Step]")
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        
        scheduler.step(vali_loss)
        
        print("\n[Test Step]")
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        print(
            f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
            f"Valid results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"MCC: {val_metrics_dict['MCC']:.5f}, "
            f"CPI: {val_metrics_dict['CPI']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f} "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"MCC: {test_metrics_dict['MCC']:.5f}, "
            f"CPI: {test_metrics_dict['CPI']:.5f}\n"
        )
        early_stopping(
            # -val_metrics_dict["CPI"],
            vali_loss,
            self.swa_model if self.swa else self.models,
            path,
        )
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + "checkpoint.pth"
    if self.swa:
        self.swa_model.load_state_dict(torch.load(best_model_path))
    else:
        self.models.load_state_dict(torch.load(best_model_path))

    return self.models