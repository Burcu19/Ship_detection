import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)


class Metrics:
    def __init__(self, y_true=None, y_pred=None):
        self.y_true = np.array(y_true) if y_true is not None else None
        self.y_pred = np.array(y_pred) if y_pred is not None else None

    def set(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def accuracy(self):
        return float(accuracy_score(self.y_true, self.y_pred))

    def precision(self):
        return float(precision_score(self.y_true, self.y_pred, zero_division=0))

    def recall(self):
        return float(recall_score(self.y_true, self.y_pred, zero_division=0))

    def f1score(self):
        return float(f1_score(self.y_true, self.y_pred, zero_division=0))

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])

    def all(self):
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1score(),
            "confusion_matrix": self.confusion_matrix().tolist(),
        }
