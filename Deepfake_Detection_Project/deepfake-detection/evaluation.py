"""
Evaluation Module for AI-Based Multimodal Deepfake Video Detection.

Provides comprehensive model evaluation metrics computation and
publication-quality visualization generation for deepfake detection models.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report as sklearn_classification_report,
)
from sklearn.preprocessing import label_binarize

from config import PLOT_DPI, PLOT_FIGSIZE, CHART_COLORS, CLASS_LABELS

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
sns.set_style("darkgrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})


# ======================================================================
# 1. compute_metrics
# ======================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray, inference_times: np.ndarray) -> dict:
    """
    Compute a comprehensive set of classification and performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (shape: (n_samples,)).
    y_pred : np.ndarray
        Predicted binary labels (shape: (n_samples,)).
    y_prob : np.ndarray
        Predicted probabilities for the positive class (shape: (n_samples,)).
    inference_times : np.ndarray
        Per-sample inference times in seconds (shape: (n_samples,)).

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, F1, AUC, MCC,
        Cohen's Kappa, specificity, sensitivity, average inference time (ms),
        and throughput (FPS).
    """
    logger.info("Computing evaluation metrics for %d samples.", len(y_true))

    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    inference_times = np.asarray(inference_times, dtype=np.float64).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        logger.warning("Cannot compute AUC — only one class present in y_true.")
        auc = float('nan')

    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(recall)  # sensitivity == recall (TPR)

    avg_inference_ms = float(np.mean(inference_times) * 1000.0)
    fps = float(1.0 / np.mean(inference_times)) if np.mean(inference_times) > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "avg_inference_time_ms": avg_inference_ms,
        "fps": fps,
    }

    logger.info("Metrics computed: Acc=%.4f  F1=%.4f  AUC=%.4f  MCC=%.4f", accuracy, f1, auc, mcc)
    return metrics


# ======================================================================
# 2. plot_confusion_matrix
# ======================================================================
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: str, labels=None) -> None:
    """
    Plot a side-by-side confusion matrix with raw counts and row-normalised
    percentages.  Saves the figure to *save_path*.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    save_path : str
        File path for saving the PNG.
    labels : list[str], optional
        Class label names.  Defaults to ``['Real', 'Fake']``.
    """
    if labels is None:
        labels = [CLASS_LABELS.get(0, "Real"), CLASS_LABELS.get(1, "Fake")]

    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    for ax, data, title, fmt in [
        (axes[0], cm, "Confusion Matrix (Counts)", "d"),
        (axes[1], cm_norm, "Confusion Matrix (Normalized)", ".2%"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
            linecolor="grey",
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle("Confusion Matrix Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


# ======================================================================
# 3. plot_accuracy_curve
# ======================================================================
def plot_accuracy_curve(history, save_path: str) -> None:
    """
    Plot training and validation accuracy vs. epochs.

    Parameters
    ----------
    history : dict or keras.History
        Object with ``history.history`` (Keras) or a plain dict containing
        ``'accuracy'`` and ``'val_accuracy'`` keys.
    save_path : str
        Destination file path for the PNG.
    """
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history

    epochs = np.arange(1, len(hist["accuracy"]) + 1)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(epochs, hist["accuracy"], marker="o", markersize=4,
            color=CHART_COLORS[0], linewidth=2, label="Training Accuracy")
    ax.plot(epochs, hist["val_accuracy"], marker="s", markersize=4,
            color=CHART_COLORS[1], linewidth=2, label="Validation Accuracy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend(loc="lower right")
    ax.set_xticks(epochs)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Accuracy curve saved to %s", save_path)


# ======================================================================
# 4. plot_loss_curve
# ======================================================================
def plot_loss_curve(history, save_path: str) -> None:
    """
    Plot training and validation loss vs. epochs.

    Parameters
    ----------
    history : dict or keras.History
        Training history object.
    save_path : str
        Destination file path for the PNG.
    """
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history

    epochs = np.arange(1, len(hist["loss"]) + 1)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(epochs, hist["loss"], marker="o", markersize=4,
            color=CHART_COLORS[0], linewidth=2, label="Training Loss")
    ax.plot(epochs, hist["val_loss"], marker="s", markersize=4,
            color=CHART_COLORS[1], linewidth=2, label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(loc="upper right")
    ax.set_xticks(epochs)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Loss curve saved to %s", save_path)


# ======================================================================
# 5. plot_roc_curve
# ======================================================================
def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """
    Plot the Receiver Operating Characteristic (ROC) curve with the AUC
    score annotated.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    save_path : str
        Destination file path for the PNG.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(fpr, tpr, color=CHART_COLORS[0], linewidth=2.5,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.2,
            label="Random Classifier")

    ax.fill_between(fpr, tpr, alpha=0.15, color=CHART_COLORS[0])

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s  (AUC=%.4f)", save_path, auc_score)


# ======================================================================
# 6. plot_precision_recall_curve
# ======================================================================
def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray,
                                save_path: str) -> None:
    """
    Plot the Precision-Recall curve with the Average Precision (AP) score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    save_path : str
        Destination file path for the PNG.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(rec, prec, color=CHART_COLORS[3], linewidth=2.5,
            label=f"PR Curve (AP = {ap:.4f})")

    # Baseline = proportion of positive class
    baseline = np.sum(y_true == 1) / len(y_true)
    ax.axhline(y=baseline, color="grey", linestyle="--", linewidth=1.2,
               label=f"Baseline (Positive Rate = {baseline:.4f})")

    ax.fill_between(rec, prec, alpha=0.12, color=CHART_COLORS[3])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left", fontsize=12)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Precision-Recall curve saved to %s  (AP=%.4f)", save_path, ap)


# ======================================================================
# 7. plot_training_history
# ======================================================================
def plot_training_history(history, save_path: str) -> None:
    """
    Combined 2×2 plot of training history:
    (a) training & validation accuracy,
    (b) training & validation loss,
    (c) learning rate schedule,
    (d) training & validation AUC (if available).

    Parameters
    ----------
    history : dict or keras.History
        Training history object.
    save_path : str
        Destination file path for the PNG.
    """
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history

    epochs = np.arange(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training History Overview", fontsize=18, fontweight="bold", y=1.01)

    # --- (a) Accuracy ---
    ax = axes[0, 0]
    ax.plot(epochs, hist["accuracy"], marker="o", markersize=3,
            color=CHART_COLORS[0], linewidth=2, label="Train Accuracy")
    ax.plot(epochs, hist["val_accuracy"], marker="s", markersize=3,
            color=CHART_COLORS[1], linewidth=2, label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # --- (b) Loss ---
    ax = axes[0, 1]
    ax.plot(epochs, hist["loss"], marker="o", markersize=3,
            color=CHART_COLORS[0], linewidth=2, label="Train Loss")
    ax.plot(epochs, hist["val_loss"], marker="s", markersize=3,
            color=CHART_COLORS[1], linewidth=2, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # --- (c) Learning Rate ---
    ax = axes[1, 0]
    lr_key = "lr" if "lr" in hist else "learning_rate"
    if lr_key in hist:
        lrs = hist[lr_key]
        ax.plot(epochs, lrs, marker="o", markersize=3,
                color=CHART_COLORS[2], linewidth=2, label="Learning Rate")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate (log scale)")
        ax.set_title("Learning Rate Schedule")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, which="both")
    else:
        ax.text(0.5, 0.5, "Learning rate data not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="grey")
        ax.set_title("Learning Rate Schedule")
        ax.set_xticks([])
        ax.set_yticks([])

    # --- (d) AUC ---
    ax = axes[1, 1]
    train_auc_key = "auc" if "auc" in hist else None
    val_auc_key = "val_auc" if "val_auc" in hist else None

    if train_auc_key is not None:
        ax.plot(epochs, hist[train_auc_key], marker="o", markersize=3,
                color=CHART_COLORS[0], linewidth=2, label="Train AUC")
    if val_auc_key is not None:
        ax.plot(epochs, hist[val_auc_key], marker="s", markersize=3,
                color=CHART_COLORS[1], linewidth=2, label="Val AUC")

    if train_auc_key or val_auc_key:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title("AUC")
        ax.legend(loc="lower right")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "AUC metric not available in history",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="grey")
        ax.set_title("AUC")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training history plot saved to %s", save_path)


# ======================================================================
# 8. plot_feature_importance
# ======================================================================
def plot_feature_importance(feature_names: list, importances: np.ndarray,
                            save_path: str, top_n: int = 20) -> None:
    """
    Horizontal bar chart of the *top_n* most important features.

    Parameters
    ----------
    feature_names : list[str]
        Names of the features.
    importances : np.ndarray
        Importance scores corresponding to *feature_names*.
    save_path : str
        Destination file path for the PNG.
    top_n : int, optional
        Number of top features to display (default 20).
    """
    feature_names = list(feature_names)
    importances = np.asarray(importances, dtype=np.float64)

    # Sort descending and pick top_n
    sorted_indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in sorted_indices]
    top_vals = importances[sorted_indices]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.45), top_n * 0.5))

    colors = plt.cm.viridis(np.linspace(0.25, 0.85, top_n))
    bars = ax.barh(range(top_n), top_vals, color=colors[::-1], edgecolor="white",
                  linewidth=0.5, height=0.7)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate bars with values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(top_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", save_path)


# ======================================================================
# 9. plot_prediction_distribution
# ======================================================================
def plot_prediction_distribution(y_prob: np.ndarray, y_true: np.ndarray,
                                 save_path: str) -> None:
    """
    Histogram with KDE overlay of prediction probabilities, coloured by
    true class (Real vs. Fake).

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities for the positive (Fake) class.
    y_true : np.ndarray
        Ground-truth binary labels.
    save_path : str
        Destination file path for the PNG.
    """
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.int64).ravel()

    real_probs = y_prob[y_true == 0]
    fake_probs = y_prob[y_true == 1]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    bins = np.linspace(0, 1, 51)
    label_real = CLASS_LABELS.get(0, "Real")
    label_fake = CLASS_LABELS.get(1, "Fake")
    color_real = CHART_COLORS[2]  # green
    color_fake = CHART_COLORS[1]  # red

    ax.hist(real_probs, bins=bins, alpha=0.55, color=color_real,
            label=f"True {label_real} (n={len(real_probs)})", density=True,
            edgecolor="white", linewidth=0.5)
    ax.hist(fake_probs, bins=bins, alpha=0.55, color=color_fake,
            label=f"True {label_fake} (n={len(fake_probs)})", density=True,
            edgecolor="white", linewidth=0.5)

    # KDE overlays
    if len(real_probs) > 1:
        from scipy.stats import gaussian_kde
        kde_real = gaussian_kde(real_probs)
        x = np.linspace(0, 1, 300)
        ax.plot(x, kde_real(x), color=color_real, linewidth=2.2, linestyle="--",
                label=f"{label_real} KDE")
    if len(fake_probs) > 1:
        from scipy.stats import gaussian_kde
        kde_fake = gaussian_kde(fake_probs)
        x = np.linspace(0, 1, 300)
        ax.plot(x, kde_fake(x), color=color_fake, linewidth=2.2, linestyle="--",
                label=f"{label_fake} KDE")

    ax.set_xlabel("Predicted Probability (Fake)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Probability Distribution by True Class")
    ax.legend(loc="upper center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Prediction distribution plot saved to %s", save_path)


# ======================================================================
# 10. plot_class_distribution
# ======================================================================
def plot_class_distribution(y: np.ndarray, save_path: str) -> None:
    """
    Side-by-side pie chart and bar chart of class distribution.

    Parameters
    ----------
    y : np.ndarray
        Array of class labels.
    save_path : str
        Destination file path for the PNG.
    """
    y = np.asarray(y, dtype=np.int64).ravel()

    classes, counts = np.unique(y, return_counts=True)
    labels = [CLASS_LABELS.get(int(c), str(c)) for c in classes]
    colors = [CHART_COLORS[2] if int(c) == 0 else CHART_COLORS[1] for c in classes]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Class Distribution", fontsize=16, fontweight="bold")

    # --- Pie chart ---
    wedges, texts, autotexts = ax_pie.pie(
        counts,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%\n(n={int(round(p / 100. * sum(counts)))})",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 12},
        pctdistance=0.55,
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
    ax_pie.set_title("Proportion", fontsize=13, fontweight="bold")

    # --- Bar chart ---
    bars = ax_bar.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.2,
                      width=0.5)
    for bar, count in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                    str(count), ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax_bar.set_xlabel("Class")
    ax_bar.set_ylabel("Count")
    ax_bar.set_title("Counts", fontsize=13, fontweight="bold")
    ax_bar.set_ylim(0, max(counts) * 1.18)
    ax_bar.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Class distribution plot saved to %s", save_path)


# ======================================================================
# 11. generate_classification_report
# ======================================================================
def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                   output_path: str,
                                   target_names=None) -> None:
    """
    Generate a scikit-learn classification report and save it as a text file.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    output_path : str
        Destination text file path.
    target_names : list[str], optional
        Class names for the report.  Defaults to ``['Real', 'Fake']``.
    """
    if target_names is None:
        target_names = [CLASS_LABELS.get(0, "Real"), CLASS_LABELS.get(1, "Fake")]

    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    report = sklearn_classification_report(y_true, y_pred, target_names=target_names)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("=" * 60 + "\n")
        fh.write("       CLASSIFICATION REPORT — Deepfake Detection\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(report)
        fh.write("\n" + "=" * 60 + "\n")

    logger.info("Classification report saved to %s", output_path)


# ======================================================================
# 12. generate_all_plots  (master function)
# ======================================================================
def generate_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    history,
    inference_times: np.ndarray,
    save_dir: str,
    feature_names: list = None,
    feature_importances: np.ndarray = None,
) -> dict:
    """
    Master function: computes all metrics, generates every plot, and
    writes the classification report.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    history : dict or keras.History
        Training history (used for accuracy/loss/history curves).
    inference_times : np.ndarray
        Per-sample inference times in seconds.
    save_dir : str
        Directory where all outputs will be saved.
    feature_names : list[str], optional
        Feature names for importance plot.
    feature_importances : np.ndarray, optional
        Feature importance scores.

    Returns
    -------
    dict
        All evaluation metrics returned by :func:`compute_metrics`.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Generating all evaluation plots in %s", save_dir)

    # --- Compute metrics ---
    metrics = compute_metrics(y_true, y_pred, y_prob, inference_times)

    # --- Individual plots ---
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(save_dir, "confusion_matrix.png"),
    )

    plot_accuracy_curve(
        history,
        os.path.join(save_dir, "accuracy_curve.png"),
    )

    plot_loss_curve(
        history,
        os.path.join(save_dir, "loss_curve.png"),
    )

    plot_roc_curve(
        y_true, y_prob,
        os.path.join(save_dir, "roc_curve.png"),
    )

    plot_precision_recall_curve(
        y_true, y_prob,
        os.path.join(save_dir, "precision_recall_curve.png"),
    )

    plot_training_history(
        history,
        os.path.join(save_dir, "training_history.png"),
    )

    plot_prediction_distribution(
        y_prob, y_true,
        os.path.join(save_dir, "prediction_distribution.png"),
    )

    plot_class_distribution(
        y_true,
        os.path.join(save_dir, "class_distribution.png"),
    )

    # --- Optional: feature importance ---
    if feature_names is not None and feature_importances is not None:
        plot_feature_importance(
            feature_names, feature_importances,
            os.path.join(save_dir, "feature_importance.png"),
        )
    else:
        logger.info("Feature importance data not provided — skipping feature importance plot.")

    # --- Classification report ---
    generate_classification_report(
        y_true, y_pred,
        os.path.join(save_dir, "classification_report.txt"),
    )

    # --- Save metrics as JSON for programmatic access ---
    import json
    metrics_path = os.path.join(save_dir, "metrics.json")
    # Convert numpy types to native Python for JSON serialization
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(serializable_metrics, fh, indent=4)
    logger.info("Metrics JSON saved to %s", metrics_path)

    logger.info("All evaluation plots and reports generated successfully in %s", save_dir)
    return metrics


# ======================================================================
# Convenience: pretty-print metrics to console
# ======================================================================
def print_metrics(metrics: dict) -> None:
    """
    Pretty-print the metrics dictionary to the logger (INFO level).

    Parameters
    ----------
    metrics : dict
        Dictionary returned by :func:`compute_metrics`.
    """
    separator = "─" * 48
    logger.info("\n%s", separator)
    logger.info("  Evaluation Metrics Summary")
    logger.info("%s", separator)
    for key, value in metrics.items():
        if "time" in key:
            logger.info("  %-30s %10.2f ms", key, value)
        elif "fps" in key:
            logger.info("  %-30s %10.2f", key, value)
        else:
            logger.info("  %-30s %10.4f", key, value)
    logger.info("%s\n", separator)


# ======================================================================
# Module-level quick-test (invoked via `python evaluation.py`)
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT if 'LOG_FORMAT' in dir() else
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    np.random.seed(42)
    n = 500
    y_true = np.random.randint(0, 2, size=n)
    y_prob = np.where(y_true == 1,
                      np.random.beta(5, 2, size=n),
                      np.random.beta(2, 5, size=n))
    y_pred = (y_prob >= 0.5).astype(int)
    inference_times = np.random.uniform(0.01, 0.05, size=n)

    fake_history = {
        "accuracy": [0.60, 0.72, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.935],
        "val_accuracy": [0.58, 0.70, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.895, 0.90],
        "loss": [0.65, 0.50, 0.40, 0.33, 0.28, 0.24, 0.21, 0.19, 0.17, 0.16],
        "val_loss": [0.68, 0.53, 0.43, 0.37, 0.33, 0.30, 0.28, 0.26, 0.25, 0.245],
        "lr": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5],
        "auc": [0.65, 0.78, 0.85, 0.89, 0.92, 0.93, 0.94, 0.95, 0.955, 0.96],
        "val_auc": [0.63, 0.76, 0.83, 0.87, 0.90, 0.91, 0.92, 0.93, 0.935, 0.94],
    }

    feature_names_demo = [f"feature_{i}" for i in range(30)]
    feature_importances_demo = np.abs(np.random.randn(30))

    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "_eval_test")

    metrics = generate_all_plots(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        history=fake_history,
        inference_times=inference_times,
        save_dir=test_dir,
        feature_names=feature_names_demo,
        feature_importances=feature_importances_demo,
    )

    print_metrics(metrics)
    print("\nAll demo plots written to:", test_dir)
