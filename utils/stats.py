"""Statistical comparison of the approaches on the test set.

A single test split gives one number per approach; with 3,669 test images, differences of
about one point can come from sampling noise alone. This module quantifies that:

- Paired bootstrap: the test images are resampled with replacement (the SAME resampled
  indices for every approach, so comparisons are paired). Gives a 95% percentile confidence
  interval for accuracy and macro F1 of each approach, and for the accuracy difference
  between each pair of approaches.
- Exact McNemar test: for each pair (A, B), only the images where exactly one of the two is
  correct matter (b = A right / B wrong, c = A wrong / B right). Under H0 (same error rate)
  b ~ Binomial(b + c, 0.5). Two-sided exact p-values, Holm-corrected for the number of pairs.

Limitation: this only measures variability due to the finite TEST set. Variability due to
training (seed, data order, augmentation) is not included.
"""

import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

N_BOOT = 10_000
ALPHA = 0.05


def load_predictions(path):
    """Read <name>_test_predictions.csv -> (y_true [N], y_pred [N])"""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return (np.array([int(r["y_true"]) for r in rows]),
            np.array([int(r["y_pred"]) for r in rows]))


def macro_f1_batch(y_true, y_pred, idx, num_classes):
    """Macro F1 for many bootstrap samples at once: idx [B, N] resampled indices -> [B].
    Averaged over all num_classes, a class with no true and no predicted samples counts
    as 0 (same as sklearn f1_score(average="macro", labels=..., zero_division=0))."""
    B = idx.shape[0]
    t, p = y_true[idx], y_pred[idx]                      # [B, N]
    offset = (np.arange(B) * num_classes)[:, None]
    n = B * num_classes
    tp = np.bincount((t + offset)[t == p], minlength=n).reshape(B, num_classes)
    support = np.bincount((t + offset).ravel(), minlength=n).reshape(B, num_classes)
    predicted = np.bincount((p + offset).ravel(), minlength=n).reshape(B, num_classes)
    denom = support + predicted                          # F1 = 2TP / (2TP + FP + FN)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return f1.mean(axis=1)


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test. Returns (b, c, p-value)."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)"""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def ci(samples):
    lo, hi = np.percentile(samples, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def significance(preds, num_classes, n_boot=N_BOOT, seed=42):
    """preds: {name: (y_true, y_pred)} with the same y_true order for every approach.
    Returns (per-approach rows, pairwise rows)."""
    names = list(preds)
    y_true = preds[names[0]][0]
    for name in names:
        assert np.array_equal(preds[name][0], y_true), f"{name}: different test order"
    N = len(y_true)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(n_boot, N))           # shared -> paired bootstrap

    correct = {k: preds[k][1] == y_true for k in names}
    boot_acc = {k: correct[k][idx].mean(axis=1) for k in names}

    per_model = []
    for k in names:
        f1 = np.concatenate([macro_f1_batch(y_true, preds[k][1], idx[i:i + 1000], num_classes)
                             for i in range(0, n_boot, 1000)])
        per_model.append({"approach": k, "accuracy": float(correct[k].mean()),
                          "acc_ci": ci(boot_acc[k]),
                          "macro_f1": float(macro_f1_batch(y_true, preds[k][1],
                                                           np.arange(N)[None], num_classes)[0]),
                          "f1_ci": ci(f1)})

    pairs = []
    for a, b in itertools.combinations(names, 2):
        n_ab, n_ba, p = mcnemar_exact(correct[a], correct[b])
        diff = boot_acc[a] - boot_acc[b]
        pairs.append({"a": a, "b": b, "acc_diff": float(correct[a].mean() - correct[b].mean()),
                      "diff_ci": ci(diff), "a_only": n_ab, "b_only": n_ba, "p": p})
    for row, p_adj in zip(pairs, holm([r["p"] for r in pairs])):
        row["p_holm"] = float(p_adj)
        row["significant"] = bool(p_adj < ALPHA)
    return per_model, pairs


# Output

def _tex(name):
    return name.replace("_", r"\_")


def _thousands(n):
    return f"{n:,}".replace(",", "{,}")


def _pval(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def save_significance(per_model, pairs, out_dir, n_boot=N_BOOT):
    """significance.json / .md / .tex in out_dir"""
    out_dir = Path(out_dir)
    (out_dir / "significance.json").write_text(
        json.dumps({"n_boot": n_boot, "alpha": ALPHA, "per_approach": per_model,
                    "pairwise": pairs}, indent=2))

    md = ["| approach | test acc [95% CI] | macro F1 [95% CI] |", "|---|---|---|"]
    md += [f"| {r['approach']} | {r['accuracy']:.3f} [{r['acc_ci'][0]:.3f}, {r['acc_ci'][1]:.3f}] "
           f"| {r['macro_f1']:.3f} [{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}] |" for r in per_model]
    md += ["", "| A vs B | acc A - B [95% CI] | only A right | only B right | McNemar p | Holm p |",
           "|---|---|---|---|---|---|"]
    md += [f"| {r['a']} vs {r['b']} | {100 * r['acc_diff']:+.1f} pp "
           f"[{100 * r['diff_ci'][0]:+.1f}, {100 * r['diff_ci'][1]:+.1f}] | {r['a_only']} | "
           f"{r['b_only']} | {r['p']:.2g} | {r['p_holm']:.2g} |" for r in pairs]
    (out_dir / "significance.md").write_text("\n".join(md) + "\n")

    tex = [
        "% Generated by `python main.py --approach stats` (utils/stats.py). Needs \\usepackage{booktabs}.",
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{Pairwise comparison on the test set (3{{,}}669 images). $\Delta$: accuracy of A minus B, "
        rf"with a 95\% paired-bootstrap CI ({_thousands(n_boot)} resamples). $n_{{A}}$/$n_{{B}}$: images only A/B "
        r"classifies correctly. $p$: exact McNemar test, Holm-corrected over the six pairs.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"A & B & $\Delta$ (pp) [95\% CI] & $n_A$ & $n_B$ & $p_\mathrm{Holm}$ \\", r"\midrule"]
    tex += [rf"{_tex(r['a'])} & {_tex(r['b'])} & ${100 * r['acc_diff']:+.1f}$ "
            rf"[${100 * r['diff_ci'][0]:+.1f}$, ${100 * r['diff_ci'][1]:+.1f}$] & {r['a_only']} & "
            rf"{r['b_only']} & {_pval(r['p_holm'])} \\" for r in pairs]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "significance.tex").write_text("\n".join(tex))
    return "\n".join(md)


def save_results_tex(rows, per_model, out_dir, n_boot=N_BOOT):
    """Main results table (results_table.tex) with bootstrap CIs on accuracy and macro F1"""
    ci_by = {r["approach"]: r for r in per_model}
    tex = [
        "% Generated by `python main.py --approach stats` (utils/stats.py). Needs \\usepackage{booktabs,graphicx}.",
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Test-set results (3{{,}}669 images). Brackets: 95\% paired-bootstrap confidence "
        rf"intervals ({_thousands(n_boot)} resamples of the test set). Train time includes validation passes; for "
        r"\texttt{features\_*} it is train-set feature extraction plus classifier fitting.}",
        r"\label{tab:results}",
        r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrccc}", r"\toprule",
        r"Approach & Trainable params & Time (s) & Train acc & Test acc [95\% CI] & Macro F1 [95\% CI] & AUC \\",
        r"\midrule"]
    for r in rows:
        c = ci_by[r["approach"]]
        tex.append(rf"{_tex(r['approach'])} & {_thousands(r['trainable_params'])} & {r['train_time_s']:.1f} & "
                   rf"{r['train_accuracy']:.3f} & {c['accuracy']:.3f} [{c['acc_ci'][0]:.3f}, {c['acc_ci'][1]:.3f}] & "
                   rf"{c['macro_f1']:.3f} [{c['f1_ci'][0]:.3f}, {c['f1_ci'][1]:.3f}] & {r['auc_roc']:.3f} \\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    (out_dir / "results_table.tex").write_text("\n".join(tex))
