import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_curve, auc
)

def evaluate_model(model, X_test, y_test, task_type, model_type):
    try:
        if model_type == "classical":
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        else:
            pred = model.predict(X_test, verbose=0)
            if task_type == "classification":
                y_pred = np.argmax(pred, axis=1)
                y_proba = pred if pred.shape[1] > 1 else None
            else:
                y_pred = pred.flatten()
                y_proba = None

        if task_type == "classification":
            return {
                "predictions": y_pred,
                "y_proba": y_proba,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            return {
                "predictions": y_pred,
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            }
    except Exception as e:
        raise RuntimeError(f"Erreur dans evaluate_model : {e}")

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Matrice de confusion")
    ax.set_ylabel("Vraies classes")
    ax.set_xlabel("Prédictions")
    return fig

def plot_regression_results(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Vraies valeurs")
    ax.set_ylabel("Prédictions")
    ax.set_title("Prédictions vs Vraies valeurs")
    return fig

def plot_training_curves(history, task_type):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].legend()
    ax[0].set_title('Loss')

    metric = 'accuracy' if task_type == "classification" else 'mae'
    ax[1].plot(history.history[metric], label=f'Train {metric}')
    ax[1].plot(history.history[f'val_{metric}'], label=f'Val {metric}')
    ax[1].legend()
    ax[1].set_title(metric.title())
    return fig

def plot_roc_curve(y_true, y_proba):
    """Courbe ROC pour la classification binaire."""
    if y_proba is not None and y_proba.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taux de faux positifs')
        ax.set_ylabel('Taux de vrais positifs')
        ax.set_title('Courbe ROC')
        ax.legend(loc="lower right")
        return fig
    return None