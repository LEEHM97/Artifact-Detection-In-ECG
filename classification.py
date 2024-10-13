import torch
import torch.nn as nn
import os
import time
import warnings
import numpy as np
import random
import Medformer

from torch import optim
from data_factory import data_provider
from utils import EarlyStopping, mcc_score

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")


class Exp_Classification():
    def __init__(self, args):
        
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args['swa']

    def _build_model(self):
        # model input depends on data
        test_data, test_loader = self._get_data(flag="VAL")
        self.args['seq_len'] = test_data.max_seq_len
        self.args['pred_len'] = 0
        self.args['enc_in'] = test_data.X.shape[2]  # redefine enc_in
        self.args['num_class'] = len(np.unique(test_data.y))
        # model init
        model = (
            Medformer.Model(self.args).float()
        )  # pass args to model
        if self.args['use_multi_gpu'] and self.args['use_gpu']:
            model = nn.DataParallel(model, device_ids=self.args['device_ids'])
        return model

    def _get_data(self, flag):
        random.seed(self.args['seed'])
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        
        return criterion

    def _select_scheduler(self, optimizer, steps):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
            
        return scheduler
    
    def _acquire_device(self):
        if self.args['use_gpu']:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args['gpu']) if not self.args['use_multi_gpu'] else self.args['devices']
            )
            device = torch.device("cuda:{}".format(self.args['gpu']))
            print("Use GPU: cuda:{}".format(self.args['gpu']))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = 0.0
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                label = label.squeeze(1).float()

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.float().cpu())

                total_loss += loss * batch_x.size(0)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = total_loss / len(vali_loader.dataset)

        preds = torch.cat(preds, 0).squeeze(1)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.sigmoid(preds)

        trues_onehot = trues.float().cpu().numpy()

        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        
        probs = probs.cpu().numpy()
        trues = torch.argmax(trues, dim=1).cpu().numpy()

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

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)

        path = (
            "./checkpoints/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args['patience'], verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim, train_steps)

        for epoch in range(self.args['train_epochs']):
            iter_count = 0
            train_loss = 0.0

            self.model.train()
            epoch_time = time.time()

            print("[Train Step]")
            for i, (batch_x, label, padding_mask) in enumerate(tqdm(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                label = label.squeeze(1).float()
                
                outputs = self.model(batch_x, padding_mask, None, None)
                
                loss = criterion(outputs, label.float())

                train_loss += loss.item() * batch_x.size(0)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = train_loss / len(train_loader.dataset)
            
            print("[Validation Step]")
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            
            scheduler.step(vali_loss)

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
            )
            
            early_stopping(
                vali_loss,
                self.swa_model if self.swa else self.model,
                path,
            )
        
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "checkpoint.pth"
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + setting
                + "/"
            )
            
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)

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
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return