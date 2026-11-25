# Homework 2: Transfer Learning on Oxford-IIIT Pet (37 breeds) with MobileNetV3-Small

This repo compares three ways to adapt an ImageNet-pre-trained **MobileNetV3-Small** to the 37 breed classes of the **Oxford-IIIT Pet** dataset:

1. **Full fine-tuning**: the whole network is trained.
2. **Classifier-head training**: the convolutional feature extractor is frozen and only the fully connected classifier is trained.
3. **Feature extraction + scikit-learn**: frozen 576-d features go into a Logistic Regression and a Linear SVM.

All three approaches use the same splits, preprocessing and evaluation protocol.

## Environment

| Component | Version |
|---|---|
| Python | 3.13.18 |
| PyTorch | 2.15.0 (+cu134) |
| TorchVision | 0.29.2 (+cu134) |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.4 |
| Pillow | 13.0.0 |
| Matplotlib | 3.11.2 |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU (driver 580.178.04) |

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu134
```

The code also runs on CPU if CUDA isn't available, just more slowly.

## Dataset

- **Source:** O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, *Cats and Dogs*, CVPR 2012. <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). Image copyright stays with the original owners.
- The images are **not stored in this repo**. To download them (about 800 MB, into `data/raw/oxford-iiit-pet/`) and write the split files, run:

```bash
python data/data.py
```

The script uses `torchvision.datasets.OxfordIIITPet` and also prints the number of images per breed in each split.

### Split protocol

| Split | Images | How it is obtained |
|---|---|---|
| train | 2,944 | 80% of the official `trainval`, stratified by breed (`train_test_split`, `random_state=42`) |
| val | 736 | the remaining 20% of `trainval` |
| test | 3,669 | official `test` partition, untouched; used only for the final evaluation |

The split files are committed in `data/splits/` (`train.csv`, `val.csv`, `test.csv`, with `image_id,label`). Per-breed counts are in `data/splits/class_counts.csv`. The classes are roughly balanced: 74–80 training images per breed, 88–100 test images. No class-balancing is applied.

### Preprocessing

- **Eval (val, test, and train when used for feature extraction or measuring train accuracy):** `MobileNet_V3_Small_Weights.DEFAULT.transforms()`, which means resize to 256, center crop to 224, then ImageNet mean/std normalization.
- **Train (approaches 1 and 2 only):** `RandomResizedCrop(224, scale=(0.5, 1.0))` and `RandomHorizontalFlip`, then the same normalization. Augmentation is applied **only** to the training set.

## Model and approaches

`torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)` (ImageNet-1K). The classifier is `Linear(576→1024) → Hardswish → Dropout(0.2) → Linear(1024→1000)`. The last layer, `classifier[3]`, is replaced by `Linear(1024→37)`. See `models/mobilenet.py`.

| Approach | Trainable parameters | What is trained |
|---|---|---|
| `full` | 1,555,781 (all) | the whole network |
| `head` | 628,773 | `classifier[0]` (Linear 576→1024) and `classifier[3]` (Linear 1024→37). `features` is frozen and its BatchNorm layers are kept in `eval()`, so their running statistics don't change |
| `features_*` | 21,349 (sklearn weights) | nothing in the CNN. Features are the 576-d output of `avgpool(features(x))`, before the classifier |

### Training setup

| | `full` | `head` | `features_logreg` | `features_linsvm` |
|---|---|---|---|---|
| Loss | CrossEntropy | CrossEntropy | multinomial log-loss | squared hinge (one-vs-rest) |
| Optimizer | AdamW, wd 1e-4, cosine LR | AdamW, wd 1e-4, cosine LR | lbfgs, `max_iter=2000` | liblinear (`dual=False`), `max_iter=5000` |
| Learning rate | 1e-4 | 1e-3 | – | – |
| Batch size | 64 | 64 | – | – |
| Epochs | 15 | 15 | – | – |
| Grid searched on val | – | – | C ∈ {0.01, 0.1, 1, 10} | C ∈ {0.001, 0.01, 0.1} |
| Model selection | epoch with best val accuracy | epoch with best val accuracy | C with best val accuracy | C with best val accuracy |
| **Selected** | epoch 15, `checkpoints/full.pt` | epoch 11, `checkpoints/head.pt` | C = 0.01 | C = 0.001 |

For the feature-based classifiers, a `StandardScaler` is fitted on the **training features only** and then applied to val and test. No dimensionality reduction is used.

## Reproducing the experiments

```bash
python data/data.py                     # 1. download the dataset and write the splits (once)
python main.py --approach all           # 2. run all three approaches, then build the table and plots
```

You can also run one approach at a time. The results table is rebuilt from every `results/*_metrics.json` that exists:

```bash
python main.py --approach full
python main.py --approach head
python main.py --approach features      # Logistic Regression and Linear SVM
python main.py --approach stats         # bootstrap CIs + McNemar tests on the saved test predictions
python models/mobilenet.py              # sanity check: output shape (1, 37) and trainable params per mode
```

## Reproducibility

- **Seeds:** `SEED = 42` for Python (`random`), NumPy and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`). It is reset before each approach. The same seed drives the stratified split and the DataLoader shuffling (`torch.Generator`). Use `--seed` to change it.
- **Determinism:** `torch.backends.cudnn.deterministic = True` and `cudnn.benchmark = False`. `torch.use_deterministic_algorithms(True)` is **not** enabled.
- **Remaining sources of variation:** some CUDA kernels (e.g. in the backward pass) are non-deterministic, and results can change across GPU models, driver/CUDA versions and library versions. Measured training times depend on the hardware and on system load.
- **No test leakage:** the test set is never used for early stopping, checkpoint selection, choosing C, or fitting the scaler. Each selected model is evaluated on it once, at the end.

## Results

Results on the test set (3,669 images). *Train acc* is measured on the training split without augmentation, using the selected model. *Train time* covers the training loop including validation passes; for `features_*` it is feature extraction on the train set plus fitting the classifier.

| Approach | Trainable params | Train time (s) | Train acc | Test acc | Macro F1 | AUC-ROC (OvR) |
|---|---|---|---|---|---|---|
| full | 1,555,781 | 95.5 | 0.960 | 0.810 | 0.807 | 0.993 |
| head | 628,773 | 83.9 | 0.995 | **0.864** | **0.863** | 0.997 |
| features_logreg | 21,349 | 15.9 | 0.988 | 0.857 | 0.855 | 0.996 |
| features_linsvm | 21,349 | 36.6 | 0.959 | 0.838 | 0.834 | 0.988 |

All saved outputs are in `results/`:

- `results.md` / `results.csv`: the table above (it also has a `loss` column; for `linsvm` that value is a softmax over SVM scores and shouldn't be read as a calibrated loss).
- `<approach>_metrics.json`: metrics, hyperparameters, per-class accuracy and the most-confused class pairs.
- `<approach>_confusion.{png,csv}`, `<approach>_per_class.csv`: confusion matrix with class names, and per-class scores.
- `<approach>_correct.png`, `<approach>_wrong.png`: the most confident correct and wrong test predictions.
- `<approach>_roc.png`, `{full,head}_history.png`: ROC curves and training/validation curves.
- `<approach>_test_predictions.csv`: per-image test predictions (row *i* = row *i* of `data/splits/test.csv`).
- `significance.{md,json,tex}`, `results_table.tex`: statistical comparison (see below) and LaTeX tables for the report.
- `comparison.png`, `per_class_accuracy.png`: comparisons across approaches.

The checkpoints for the selected models are `checkpoints/full.pt` and `checkpoints/head.pt` (about 6 MB each, as a `state_dict`).

## Statistical comparison

A single test split gives one number per approach, so `python main.py --approach stats` (`utils/stats.py`) checks which differences are larger than test-set sampling noise:

- **95% paired-bootstrap CIs** (10,000 resamples of the 3,669 test images, seed 42; the same resamples for every approach) for accuracy, macro F1, and the accuracy difference of each pair.
- **Exact McNemar test** on the images where exactly one of the two approaches is correct, Holm-corrected over the 6 pairs.

| A vs B | Δ acc [95% CI] | Holm p |
|---|---|---|
| head vs features_logreg | +0.7 pp [−0.1, +1.5] | 0.11 |
| head vs features_linsvm | +2.6 pp [+1.5, +3.6] | < 0.001 |
| features_logreg vs features_linsvm | +1.9 pp [+1.0, +2.8] | < 0.001 |
| full vs each other approach | −2.8 to −5.4 pp | < 0.001 |

**head** and **features_logreg** cannot be told apart on this test set. Every other difference is significant. This only covers variability from the finite test set: training variability (seeds, data order) is not measured. If predictions files are missing (older runs), the stats step regenerates them from the saved checkpoints and selected C and checks that they reproduce the saved accuracies exactly.

## Repository layout

```
data/data.py          download, stratified split, transforms, Dataset/DataLoaders
data/splits/          split files and per-breed counts (committed)
models/mobilenet.py   model construction per approach, freezing, feature extraction
utils/train.py        training loop with checkpoint selection on validation
utils/metrics.py      predictions, accuracy / macro F1 / AUC / confusion, saving tables
utils/plots.py        confusion matrices, curves, prediction examples, comparisons
utils/stats.py        bootstrap confidence intervals, McNemar tests, LaTeX tables
main.py               runs the approaches and produces results/
results/              generated metrics and figures
checkpoints/          selected PyTorch checkpoints
```

## Reference

O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. *Cats and Dogs.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
