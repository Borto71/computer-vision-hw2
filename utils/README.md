# utils/

- `train.py`: training loop (AdamW + cosine schedule) that keeps the checkpoint with the best validation accuracy.
- `metrics.py`: predictions, accuracy / macro F1 / AUC / confusion matrix, saving metrics, per-image predictions and the results table.
- `plots.py`: confusion matrices, training curves, ROC curves, prediction examples, comparison plots.
- `stats.py`: paired-bootstrap confidence intervals, exact McNemar tests (Holm-corrected) and the LaTeX tables in `results/`.
