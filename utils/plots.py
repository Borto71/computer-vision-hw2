from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, only files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Traing

def plot_history(history, path, title=None):
    """Loss and accuracy per epoch, train vs val.
    history = {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]}
    optional "best_epoch": int (1-based) is marked with a vertical line."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))

    for ax, key, label in ((ax_l, "loss", "Cross-entropy loss"), (ax_a, "acc", "Accuracy")):
        for split in ("train", "val"):
            if f"{split}_{key}" in history:
                ax.plot(epochs, history[f"{split}_{key}"], marker="o", markersize=3, label=split)
        if history.get("best_epoch"):
            ax.axvline(history["best_epoch"], linestyle="--", color="gray",
                       label=f"selected (epoch {history['best_epoch']})")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(label)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)

# Evaluation

def plot_confusion_matrix(cm, classes, path, normalize=True, title=None):
    """Confusion matrix with class names. normalize=True -> each row sums to 1
    (readable even if classes have different sizes)."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    n = len(classes)
    fig, ax = plt.subplots(figsize=(0.32 * n + 3, 0.32 * n + 2))
    im = ax.imshow(cm, vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Fraction of true class" if normalize else "Images")
    ax.set_xticks(range(n), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n), classes, fontsize=7)
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)

def plot_roc_curves(y_true, y_prob, classes, path, title=None):
    """One-vs-rest ROC: one thin line per class, plus micro- and macro-average."""
    n = len(classes)
    y_bin = label_binarize(y_true, classes=np.arange(n))
    y_prob = np.asarray(y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    grid = np.linspace(0, 1, 501)
    tprs = []
    for c in range(n):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        tprs.append(np.interp(grid, fpr, tpr))
        ax.plot(fpr, tpr, color="gray", alpha=0.25, linewidth=0.8,
                label="per class" if c == 0 else None)

    macro_tpr = np.mean(tprs, axis=0)
    macro_tpr[0] = 0.0
    ax.plot(grid, macro_tpr, linewidth=2, label=f"macro-average (AUC {auc(grid, macro_tpr):.3f})")
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    ax.plot(fpr, tpr, linewidth=2, label=f"micro-average (AUC {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    _save(fig, path)

def plot_per_class_accuracy(acc_by_approach, classes, path, title=None):
    """Horizontal grouped bars, one group per breed, one bar per approach.
    acc_by_approach = {"full": array[C], "head": array[C], ...}
    Breeds are sorted by the accuracy of the first approach (worst at the top)."""
    names = list(acc_by_approach)
    first = np.asarray(acc_by_approach[names[0]])
    order = np.argsort(first)[::-1]  # barh draws bottom-up -> worst at the top
    n, k = len(classes), len(names)
    height = 0.8 / k

    fig, ax = plt.subplots(figsize=(8, 0.28 * n * max(1, k / 2) + 1.5))
    y = np.arange(n)
    for i, name in enumerate(names):
        acc = np.asarray(acc_by_approach[name])[order]
        ax.barh(y - (i - (k - 1) / 2) * height, acc, height=height, label=name)
    ax.set_yticks(y, [classes[j] for j in order], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-class accuracy")
    ax.set_ylim(-0.6, n - 0.4)
    if k > 1:
        ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)

def plot_comparison(rows, path, metrics=("accuracy", "macro_f1", "auc_roc"), title=None):
    """Grouped bars of the test metrics of every approach.
    rows = [{"approach": "full", "accuracy": .., "macro_f1": .., ...}, ...]"""
    names = [r["approach"] for r in rows]
    x = np.arange(len(metrics))
    width = 0.8 / len(rows)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(metrics) * len(rows) / 2), 4))
    for i, r in enumerate(rows):
        vals = [r[m] for m in metrics]
        bars = ax.bar(x + (i - (len(rows) - 1) / 2) * width, vals, width, label=names[i])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, [m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test score")
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    _save(fig, path)

# Examples

def pick_examples(y_true, y_prob, n=8):
    """Indices of representative predictions: the n most confident correct ones and the
    n most confident wrong ones (the most instructive errors)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    conf = y_prob.max(axis=1)
    correct = np.where(y_pred == y_true)[0]
    wrong = np.where(y_pred != y_true)[0]
    return (correct[np.argsort(-conf[correct])][:n].tolist(),
            wrong[np.argsort(-conf[wrong])][:n].tolist())

def plot_predictions(dataset, indices, y_true, y_prob, classes, path, ncols=4, title=None,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Grid of images with true / predicted breed and confidence.
    `dataset` must return normalized tensors (eval transforms), `indices` index into it
    and into y_true / y_prob (same order, i.e. a loader with shuffle=False)."""
    if not indices:
        return
    y_prob = np.asarray(y_prob)
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    nrows = int(np.ceil(len(indices) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), indices):
        img, _ = dataset[idx]
        img = np.clip(np.asarray(img) * std + mean, 0, 1).transpose(1, 2, 0)
        pred = int(y_prob[idx].argmax())
        ok = pred == int(y_true[idx])
        ax.imshow(img)
        ax.set_title(f"true: {classes[y_true[idx]]}\npred: {classes[pred]} ({y_prob[idx, pred]:.2f})",
                     fontsize=8, color="green" if ok else "red")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
