"""Homework 2 - Transfer learning on Oxford-IIIT Pet (37 breeds) with MobileNetV3-Small.

Usage (from the repository root):
    python data/data.py                     # download the dataset and write the splits (once)
    python main.py --approach all           # full fine-tuning, head training, features + sklearn
"""

import argparse
import random
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision

from data.data import get_classes, get_loaders, make_splits

# Parameters set-up
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4

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
    p.add_argument("--approach", choices=["full", "head", "features", "all"], default="all")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    for d in (RES_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | "
          f"sklearn {sklearn.__version__} | device {device}")

    make_splits()  # same seed -> same files, safe to call every time
    loaders = get_loaders(BATCH_SIZE, NUM_WORKERS, args.seed)
    classes = get_classes()
    print(f"{len(classes)} classes | " +
          ", ".join(f"{k} {len(v.dataset)}" for k, v in loaders.items()))

    # TODO: the three approaches go here (models/, utils/)


if __name__ == "__main__":
    main()
