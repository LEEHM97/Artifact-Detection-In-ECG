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

warnings.filterwarnings("ignore")

class EnsembleModel(nn.Module):
    def __init__(self, args, models):
        super(EnsembleModel, self).__init__()
        
        self.args = args
        self.models = nn.ModuleList(models)
        self.device = 'cuda:0'
        
    def forward(self, x, padding_mask, a, b):
        outputs = [models(x, padding_mask, a, b) for models in self.models]
        outputs = torch.mean(torch.stack(outputs), dim=0)
        
        return outputs
    
    def _get_data(self, flag):
        random.seed(self.args['seed'])
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.models.parameters(), lr=self.args['learning_rate'], weight_decay=self.args['wd'])
        model_optim = optim.Adam(self.models.parameters(), lr=self.args['learning_rate'])
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss()
        return criterion
    
    def _select_scheduler(self, optimizer):
        # scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=CONFIG['T_0'], T_mult=CONFIG['T_mult'], 
                                            #   eta_max=CONFIG['max_lr'], T_up=CONFIG['T_up'], gamma=CONFIG['gamma'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        return scheduler
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        
        self.models.eval()
        
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.forward(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                label_ = nn.functional.one_hot(label.long(), num_classes=2)
                
                loss = criterion(pred, label_.float().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args['num_class'],
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)

        f1 = f1_score(trues, predictions, average="binary")
        auroc = roc_auc_score(trues_onehot, probs)
        mcc = mcc_score(trues, predictions)

        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="binary"),
            "Recall": recall_score(trues, predictions, average="binary"),
            "F1": f1,
            "AUROC": auroc,
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
            "MCC": mcc,
            "CPI": (0.25 * f1) + (0.25 * auroc) + (0.5 * mcc)
        }

        self.models.train()

        return total_loss, metrics_dict

    def train(self, setting):
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
                self.models,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "checkpoint.pth"
        
        self.models.load_state_dict(torch.load(best_model_path))

        return self.models

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading models")
            path = (
                "./checkpoints/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No models found at %s" % model_path)
            
            self.models.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
            "./results/"
            + self.args['task_name']
            + "/"
            + self.args['model_id']
            + "/"
            + self.args['model']
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
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
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f} ,"
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"MCC: {test_metrics_dict['MCC']:.5f}, "
            f"CPI: {test_metrics_dict['CPI']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
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
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f} ,"
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"MCC: {test_metrics_dict['MCC']:.5f}, "
            f"CPI: {test_metrics_dict['CPI']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return    