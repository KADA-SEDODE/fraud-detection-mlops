import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    classification_report,
)

from src.logging_utils import get_logger

logger = get_logger(__name__)



#  Calcul des métriques

def compute_metrics(y_true, y_proba, threshold=0.5) -> dict:
    """
    Calcule les principales métriques de classification binaire.
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "auprc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    logger.info(
        "Scores - ROC_AUC: %.4f | AUPRC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
        metrics["roc_auc"],
        metrics["auprc"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )

    logger.info("Classification report (seuil = %.2f):\n%s",
                threshold, classification_report(y_true, y_pred, digits=4))

    return metrics



#  Courbes de performance

def plot_roc_curve(y_true, y_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Courbe ROC sauvegardée dans {save_path}")
    else:
        plt.show()


def plot_precision_recall_curve(y_true, y_proba, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPRC = {auc_pr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Precision-Recall")
    plt.legend(loc="lower left")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Courbe Precision-Recall sauvegardée dans {save_path}")
    else:
        plt.show()



#  Fonction d'évaluation complète

def evaluate_model(model, X_test, y_test, threshold=0.5, plot=False, save_dir=None):
    """
    Calcule les métriques et (optionnellement) génère les courbes.
    """
    logger.info("Évaluation complète du modèle...")
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_proba, threshold)

    if plot or save_dir:
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            roc_path = f"{save_dir}/roc_curve.png"
            pr_path = f"{save_dir}/precision_recall_curve.png"
        else:
            roc_path = pr_path = None

        plot_roc_curve(y_test, y_proba, save_path=roc_path)
        plot_precision_recall_curve(y_test, y_proba, save_path=pr_path)

    return metrics


if __name__ == "__main__":
    logger.info("Module metrics prêt pour utilisation dans train.py et predict.py")
