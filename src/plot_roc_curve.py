import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

def plot_roc_auc(y_true, y_pred_proba, n_classes=None):
    """
    Plots ROC curve and prints AUC score.
    Args:
        y_true: True labels (binary or one-hot encoded for multi-class)
        y_pred_proba: Predicted probabilities (shape: [n_samples, n_classes] for multi-class)
        n_classes: Number of classes (for multi-class), optional
    """
    if n_classes is None:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"AUC Score: {auc_score:.4f}")
    else:
        # Multi-class (one-vs-rest)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            auc_score = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Multi-class)')
        plt.legend()
        plt.grid(True)
        plt.show()