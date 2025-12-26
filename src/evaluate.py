"""Evaluation helpers: metrics and simple profit calculation."""
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def evaluate_predictions(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
    }


def profit_for_threshold(y_true, y_proba, threshold, profit_per_success=100.0, cost_per_contact=10.0):
    preds = (y_proba >= threshold).astype(int)
    contacts = preds.sum()
    successes = ((preds == 1) & (y_true == 1)).sum()
    profit = successes * profit_per_success - contacts * cost_per_contact
    return profit
