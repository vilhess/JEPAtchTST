import torch 
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

class StreamAUC:
    def __init__(self):
        self.test_scores = []
        self.test_labels = []
    
    def update(self, errors, labels):

        self.test_scores.append(errors)
        self.test_labels.append(labels)
    
    def compute(self):
        self.test_scores = torch.cat(self.test_scores).detach().cpu().numpy()
        self.test_labels = torch.cat(self.test_labels).detach().cpu().numpy()

        auc = roc_auc_score(y_true=self.test_labels, y_score=self.test_scores)
        return auc
    
    def reset(self):
        self.test_scores = []
        self.test_labels = []

class StreamL2Loss:
    def __init__(self):
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets):
        self.predictions.append(preds)
        self.targets.append(targets)
    
    def compute(self):
        self.predictions = torch.cat(self.predictions).detach().cpu()
        self.targets = torch.cat(self.targets).detach().cpu()

        loss = F.mse_loss(self.predictions, self.targets)
        return loss.item()
    
    def reset(self):
        self.predictions = []
        self.targets = []

class StreamAccuracy:
    def __init__(self):
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets):
        self.predictions.extend(preds.tolist())
        self.targets.extend(targets.tolist())
    
    def compute(self):
        correct = sum(p == t for p, t in zip(self.predictions, self.targets))
        acc = correct / len(self.targets)
        return acc
    
    def reset(self):
        self.predictions = []
        self.targets = []