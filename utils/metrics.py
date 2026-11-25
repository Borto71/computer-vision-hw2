"""Evaluation metrics shared by the three approaches.

Every approach ends up with the same two arrays, so all metrics are computed the same way:
    y_true [N]     integer labels 0..C-1
    y_prob [N, C]  class probabilities (softmax of the logits, or predict_proba of sklearn)

Metrics required by the homework: accuracy, macro F1, confusion matrix with class names.
Extra: macro AUC-ROC (one-vs-rest), per-class accuracy, cross-entropy loss.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             roc_auc_score)

# Prediction

@torch.no_grad()
def predict(model, loader, device, criterion=None):
    """Run a PyTorch model over a loader.
    Returns (y_true [N], y_prob [N, C], mean loss or None)."""
    model.eval()
    labels, probs = [], []
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * y.size(0)
            n += y.size(0)
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(y.cpu())
    loss = total_loss / n if criterion is not None else None
    return torch.cat(labels).numpy(), torch.cat(probs).numpy(), loss

def scores_to_proba(scores):
    """Softmax for classifiers without predict_proba (e.g. LinearSVC.decision_function),
    so that AUC-ROC and loss can still be computed. Ranking (= AUC) is unchanged."""
    scores = np.asarray(scores, dtype=np.float64)
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# Metrics

def per_class_accuracy(cm):
    """Fraction of images of each class predicted correctly (= per-class recall).
    Classes with no samples get NaN."""
    cm = np.asarray(cm)
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(support > 0, np.diag(cm) / support, np.nan)

def compute_metrics(y_true, y_prob, num_classes=None):
    """All metrics from labels and probabilities. Returns a dict (arrays as numpy)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    num_classes = num_classes or y_prob.shape[1]
    labels = np.arange(num_classes)
    y_pred = y_prob.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
    except ValueError:  # a class missing from y_true (e.g. a tiny debug subset)
        auc = float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "auc_roc": auc,
        "loss": log_loss(y_true, y_prob, labels=labels),  # mean cross-entropy, same for all approaches
        "per_class_accuracy": per_class_accuracy(cm),
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }

def most_confused(cm, classes, k=10):
    """Top-k (true, predicted, count) off-diagonal pairs of the confusion matrix"""
    cm = np.asarray(cm).copy()
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    pairs = []
    for idx in flat:
        t, p = np.unravel_index(idx, cm.shape)
        if cm[t, p] == 0:
            break
        pairs.append((classes[t], classes[p], int(cm[t, p])))
    return pairs

def summary(metrics):
    """Only the scalar metrics, ready for printing or a results table"""
    return {k: float(metrics[k]) for k in ("accuracy", "macro_f1", "auc_roc", "loss")}


def format_summary(metrics, prefix=""):
    s = summary(metrics)
    return (f"{prefix}acc {s['accuracy']:.4f} | macro F1 {s['macro_f1']:.4f} | "
            f"AUC {s['auc_roc']:.4f} | loss {s['loss']:.4f}")

# Saving

def save_metrics(metrics, classes, out_dir, name, extra=None):
    """Write <name>_metrics.json, <name>_per_class.csv and <name>_confusion.csv.
    `extra` is merged in the json (trainable params, training time, hyperparameters...)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = summary(metrics)
    data["per_class_accuracy"] = {c: _nan_to_none(a) for c, a in
                                  zip(classes, metrics["per_class_accuracy"])}
    data["most_confused"] = most_confused(metrics["confusion_matrix"], classes)
    if extra:
        data.update(extra)
    with open(out_dir / f"{name}_metrics.json", "w") as f:
        json.dump(data, f, indent=2)

    cm = metrics["confusion_matrix"]
    with open(out_dir / f"{name}_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "breed", "support", "correct", "accuracy"])
        for i, c in enumerate(classes):
            w.writerow([i, c, int(cm[i].sum()), int(cm[i, i]), metrics["per_class_accuracy"][i]])

    with open(out_dir / f"{name}_confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred", *classes])
        for c, row in zip(classes, cm):
            w.writerow([c, *row.tolist()])

def save_predictions(y_true, y_prob, out_dir, name):
    """Per-image test predictions -> <name>_test_predictions.csv (row i = row i of
    data/splits/test.csv, the test loader is not shuffled). Used by utils/stats.py."""
    y_prob = np.asarray(y_prob)
    with open(Path(out_dir) / f"{name}_test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "y_true", "y_pred", "confidence"])
        for i, (t, p) in enumerate(zip(y_true, y_prob)):
            w.writerow([i, int(t), int(p.argmax()), f"{p.max():.6f}"])

def save_results_table(rows, path):
    """Comparison table of all approaches (one dict per row) as csv + markdown.
    Suggested keys: approach, trainable_params, train_time_s, train_acc, test accuracy,
    macro_f1, auc_roc, loss."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))  # union, keeps order

    with open(path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, int):
            return f"{v:,}"
        return "" if v is None else str(v)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r.get(k)) for k in keys) + " |" for r in rows]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")

def _nan_to_none(x):
    return None if np.isnan(x) else float(x)
