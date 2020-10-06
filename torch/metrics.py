#################################################################################################################
# The Pytorch tool module
# Implementation of the metrics function
# Author: Aya Saad
# email: aya.saad@ntnu.no
#
# Date created: 6 April 2020
#
# Project: AILARON
# Contact
# email: annette.stahl@ntnu.no
# funded by RCN IKTPLUSS program (project number 262741) and supported by NTNU AMOS
# Copyright @NTNU 2020
#######################################
import torch
from sklearn import metrics

def METRIX(y_true: torch.Tensor, y_pred: torch.Tensor,is_training=False) -> torch.Tensor:
    '''Calculate Accuracy, Recall, Precision and F1 score. Can work with gpu tensors
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    y_true = y_true.to("cpu")
    y_pred = y_pred.to("cpu")
    return metrics.accuracy_score(y_true, y_pred),\
           metrics.balanced_accuracy_score(y_true, y_pred),\
           metrics.precision_score(y_true,y_pred, average='weighted'),\
           metrics.recall_score(y_true, y_pred, average='weighted'),\
           metrics.f1_score(y_true, y_pred, average='weighted'),\
           metrics.classification_report(y_true, y_pred, digits=3)


