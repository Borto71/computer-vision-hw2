"""Homework 2 - Transfer learning on Oxford-IIIT Pet (37 breeds) with MobileNetV3-Small.

Usage (from the repository root):
    python data/data.py                     # download the dataset and write the splits (once)
    python main.py --approach all           # full fine-tuning, head training, features + sklearn
    python main.py --approach full          # only one approach (the results table uses all saved runs)
    python main.py --approach stats         # only bootstrap CIs + McNemar tests on the saved predictions
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data.data import get_classes, get_loaders, make_splits
from models.mobilenet import build_model, count_params, extract_features
from utils.metrics import (compute_metrics, format_summary, predict, save_metrics,
                           save_predictions, save_results_table, scores_to_proba)
from utils.plots import (pick_examples, plot_comparison, plot_confusion_matrix, plot_history,
                         plot_per_class_accuracy, plot_predictions, plot_roc_curves)
from utils.stats import (N_BOOT, load_predictions, save_results_tex, save_significance,
                         significance)
from utils.train import fit

# Parameters set-up
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 15
FULL_LR = 1e-4                    # small: do not destroy the pre-trained weights
HEAD_LR = 1e-3                    # larger: only the classifier learns
C_VALUES = [0.01, 0.1, 1, 10]     # regularization strengths tried on the validation set (logreg)
SVM_C_VALUES = [0.001, 0.01, 0.1] # SVM: large C converges very slowly and scores worse

# Paths
ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"


def set_seed(seed):
    """Same seed for Python, NumPy and PyTorch (CPU and GPU)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--approach", choices=["full", "head", "features", "stats", "all"], default="all")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def evaluate(name, y_true, y_prob, extra, classes, test_set):
    """Test metrics + plots of one approach, everything saved in results/"""
    m = compute_metrics(y_true, y_prob)
    print(format_summary(m, f"  TEST {name}: "))
    save_metrics(m, classes, RES_DIR, name, extra=extra)
    save_predictions(y_true, y_prob, RES_DIR, name)  # for the significance tests

    plot_confusion_matrix(m["confusion_matrix"], classes, RES_DIR / f"{name}_confusion.png", title=name)
    plot_roc_curves(y_true, y_prob, classes, RES_DIR / f"{name}_roc.png", title=name)
    correct, wrong = pick_examples(y_true, y_prob, n=8)
    plot_predictions(test_set, correct, y_true, y_prob, classes,
                     RES_DIR / f"{name}_correct.png", title=f"{name}: most confident correct")
    plot_predictions(test_set, wrong, y_true, y_prob, classes,
                     RES_DIR / f"{name}_wrong.png", title=f"{name}: most confident errors")


def run_torch(mode, lr, loaders, eval_loaders, device, classes):
    """Approach 1 (mode='full') and 2 (mode='head')"""
    print(f"\n=== {mode} ===")
    set_seed(SEED)  # every approach starts from the same random state
    model = build_model(mode).to(device)
    trainable, total = count_params(model)
    print(f"  trainable parameters {trainable:,} / {total:,}")

    history, train_time = fit(model, mode, loaders, device, lr, EPOCHS, CKPT_DIR / f"{mode}.pt")
    plot_history(history, RES_DIR / f"{mode}_history.png", title=mode)

    # train accuracy of the selected checkpoint, without augmentation (for the train/test gap)
    y_tr, p_tr, _ = predict(model, eval_loaders["train"], device)
    train_acc = compute_metrics(y_tr, p_tr)["accuracy"]

    # the test set is used only here, once, after model selection
    y_true, y_prob, _ = predict(model, loaders["test"], device)
    extra = {"trainable_params": trainable, "train_time_s": train_time, "train_accuracy": train_acc,
             "best_epoch": history["best_epoch"], "epochs": EPOCHS, "lr": lr,
             "batch_size": BATCH_SIZE, "optimizer": "AdamW (wd 1e-4) + cosine schedule",
             "loss_fn": "CrossEntropyLoss", "checkpoint": f"checkpoints/{mode}.pt"}
    evaluate(mode, y_true, y_prob, extra, classes, loaders["test"].dataset)

def run_features(eval_loaders, device, classes):
    """Approach 3: frozen MobileNet features (576-d after pooling) + scikit-learn classifier"""
    print("\n=== features ===")
    model = build_model("features").to(device)

    feats, extract_time = {}, {}
    for split in ("train", "val", "test"):
        start = time.time()
        feats[split] = extract_features(model, eval_loaders[split], device)
        extract_time[split] = time.time() - start
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = feats["train"], feats["val"], feats["test"]
    print(f"  features {X_tr.shape[1]}-d, train extraction {extract_time['train']:.1f}s")

    scaler = StandardScaler().fit(X_tr)  # fitted on train only
    X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    classifiers = {
        "logreg": (C_VALUES, lambda C: LogisticRegression(C=C, max_iter=2000),
                   lambda clf, X: clf.predict_proba(X)),
        # dual=False: faster when samples (2944) > features (576)
        "linsvm": (SVM_C_VALUES, lambda C: LinearSVC(C=C, dual=False, max_iter=5000),
                   lambda clf, X: scores_to_proba(clf.decision_function(X))),
    }
    for name, (c_values, make, get_proba) in classifiers.items():
        best = None
        for C in c_values:  # choose C on the validation set
            start = time.time()
            clf = make(C).fit(X_tr, y_tr)
            fit_time = time.time() - start
            val_acc = clf.score(X_va, y_va)
            print(f"  {name} C={C:<5} val acc {val_acc:.4f}")
            if best is None or val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "C": C, "clf": clf, "fit_time": fit_time}

        clf = best["clf"]
        print(f"  {name}: selected C={best['C']}")
        extra = {"trainable_params": int(clf.coef_.size + clf.intercept_.size),  # CNN frozen
                 "train_time_s": extract_time["train"] + best["fit_time"],
                 "train_accuracy": clf.score(X_tr, y_tr), "C": best["C"],
                 "scaling": "StandardScaler fitted on train"}
        evaluate(f"features_{name}", y_te, get_proba(clf, X_te), extra, classes,
                 eval_loaders["test"].dataset)

APPROACHES = ["full", "head", "features_logreg", "features_linsvm"]


def predictions_from_saved(name, eval_loaders, device):
    """Test predictions of an already selected model, WITHOUT re-training or re-selecting:
    full/head load their checkpoint, features_* refit the classifier with the C stored in
    <name>_metrics.json (lbfgs/liblinear are deterministic). Used only when
    <name>_test_predictions.csv is missing (runs made before predictions were saved)."""
    d = json.loads((RES_DIR / f"{name}_metrics.json").read_text())
    if name in ("full", "head"):
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(CKPT_DIR / f"{name}.pt", map_location=device))
        y_true, y_prob, _ = predict(model, eval_loaders["test"], device)
    else:
        model = build_model("features").to(device)
        X_tr, y_tr = extract_features(model, eval_loaders["train"], device)
        X_te, y_true = extract_features(model, eval_loaders["test"], device)
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
        if name == "features_logreg":
            y_prob = LogisticRegression(C=d["C"], max_iter=2000).fit(X_tr, y_tr).predict_proba(X_te)
        else:
            clf = LinearSVC(C=d["C"], dual=False, max_iter=5000).fit(X_tr, y_tr)
            y_prob = scores_to_proba(clf.decision_function(X_te))
    acc = float((np.asarray(y_prob).argmax(1) == np.asarray(y_true)).mean())
    if abs(acc - d["accuracy"]) > 1e-9:
        raise RuntimeError(f"{name}: regenerated test acc {acc:.4f} != saved {d['accuracy']:.4f}")
    save_predictions(y_true, y_prob, RES_DIR, name)


def run_stats(eval_loaders, device, classes):
    """Bootstrap confidence intervals and pairwise McNemar tests on the test predictions"""
    print("\n=== stats ===")
    names = [n for n in APPROACHES if (RES_DIR / f"{n}_metrics.json").exists()]
    for name in names:
        if not (RES_DIR / f"{name}_test_predictions.csv").exists():
            print(f"  {name}: regenerating test predictions from the selected model")
            predictions_from_saved(name, eval_loaders, device)
    if len(names) < 2:
        return
    preds = {n: load_predictions(RES_DIR / f"{n}_test_predictions.csv") for n in names}
    per_model, pairs = significance(preds, len(classes), n_boot=N_BOOT, seed=SEED)
    print(save_significance(per_model, pairs, RES_DIR))

    rows = [json.loads((RES_DIR / f"{n}_metrics.json").read_text()) | {"approach": n} for n in names]
    save_results_tex(rows, per_model, RES_DIR)


def make_table(classes):
    """Results table + comparison plots from every *_metrics.json saved so far"""
    rows, per_class = [], {}
    for name in APPROACHES:
        path = RES_DIR / f"{name}_metrics.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        rows.append({"approach": name, "trainable_params": d["trainable_params"],
                     "train_time_s": round(d["train_time_s"], 1),
                     "train_accuracy": d["train_accuracy"], "accuracy": d["accuracy"],
                     "macro_f1": d["macro_f1"], "auc_roc": d["auc_roc"], "loss": d["loss"]})
        per_class[name] = np.array([d["per_class_accuracy"][c] for c in classes], dtype=float)
    if not rows:
        return

    save_results_table(rows, RES_DIR / "results")
    plot_comparison(rows, RES_DIR / "comparison.png", title="Test metrics")
    plot_per_class_accuracy(per_class, classes, RES_DIR / "per_class_accuracy.png")
    print("\n" + (RES_DIR / "results.md").read_text())

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True  # reproducible convolutions (a bit slower)
    torch.backends.cudnn.benchmark = False
    for d in (RES_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | "
          f"sklearn {sklearn.__version__} | device {device}")

    make_splits()  # same seed -> same files, safe to call every time
    loaders = get_loaders(BATCH_SIZE, NUM_WORKERS, args.seed)                      # train with augmentation
    eval_loaders = get_loaders(BATCH_SIZE, NUM_WORKERS, args.seed, augment=False)  # train without
    classes = get_classes()
    print(f"{len(classes)} classes | " +
          ", ".join(f"{k} {len(v.dataset)}" for k, v in loaders.items()))

    if args.approach in ("full", "all"):
        run_torch("full", FULL_LR, loaders, eval_loaders, device, classes)
    if args.approach in ("head", "all"):
        run_torch("head", HEAD_LR, loaders, eval_loaders, device, classes)
    if args.approach in ("features", "all"):
        run_features(eval_loaders, device, classes)

    if args.approach != "stats":
        make_table(classes)
    if args.approach in ("stats", "all"):
        run_stats(eval_loaders, device, classes)

if __name__ == "__main__":
    main()
