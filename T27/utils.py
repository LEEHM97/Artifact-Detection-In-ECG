import numpy as np
import torch
import matplotlib.pyplot as plt
import math
plt.switch_backend("agg")


def adjust_learning_rate(optimizer, epoch, args, scheduler=None):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    elif args.lradj == "onecycle":
        if scheduler is not None:
            scheduler.step()  # Update the learning rate using OneCycleLR
            lr = scheduler.get_last_lr()[0]  # Get the current learning rate
            print("Updating learning rate to {}".format(lr))
            return  # Early return since scheduler handles the update

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.cpi_min = 0
        self.delta = delta

    # def __call__(self, val_loss, model, path):
    #     score = -val_loss
    #     if self.best_score is None:
    #         self.best_score = score
    #         self.save_checkpoint(val_loss, model, path)
    #     elif score < self.best_score + self.delta:
    #         self.counter += 1
    #         print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
    #         if self.counter >= self.patience:
    #             self.early_stop = True
    #     else:
    #         self.best_score = score
    #         self.save_checkpoint(val_loss, model, path)
    #         self.counter = 0
            
    def __call__(self, cpi, model, path):
        score = cpi
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(cpi, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(cpi, model, path)
            self.counter = 0

    def save_checkpoint(self, cpi, model, path):
        if self.verbose:
            print(
                f"Metric score decreased ({self.cpi_min:.6f} --> {cpi:.6f}).  Saving model ...\n"
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.cpi_min = cpi


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def mcc_score(y_true, y_pred):
    # -1과 1사이의 값을 가지며, 1에 가까울수록 비슷
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def calculate_cpi(y_true, y_pred, f1, auroc):    
    # MCC 계산
    mcc_value = mcc_score(y_true, y_pred)
    
    # CPI 계산
    cpi = 0.25 * f1 + 0.25 * auroc + 0.5 * mcc_value
    
    return cpi