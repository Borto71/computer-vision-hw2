import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights

# Parameters set-up
SEED = 42
VAL_FRAC = 0.2     # fraction of the official trainval that becomes validation (80% train, 20% val)
IMG_SIZE = 224     # input size of MobileNetV3-Small

# Paths
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"                  # torchvision puts everything in raw/oxford-iiit-pet/
PET_DIR = RAW_DIR / "oxford-iiit-pet"
SPLIT_DIR = HERE / "splits"             # small csv files, these go on git


def download():
    """Download the official trainval and test partitions (only the first time)"""
    trainval = OxfordIIITPet(RAW_DIR, split="trainval", target_types="category", download=True)
    test = OxfordIIITPet(RAW_DIR, split="test", target_types="category", download=True)
    return trainval, test


def read_official(split):
    """Read annotations/<split>.txt -> list of (image_id, label 0..36)"""
    rows = []
    with open(PET_DIR / "annotations" / f"{split}.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            image_id, class_id, _species, _breed_id = line.split()
            rows.append((image_id, int(class_id) - 1))  # file labels start from 1
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label"])
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return [(r["image_id"], int(r["label"])) for r in csv.DictReader(f)]


def make_splits(seed=SEED, val_frac=VAL_FRAC):
    """Split the official trainval into train/val stratified by breed, keep the official test as is"""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    trainval = read_official("trainval")
    test = read_official("test")

    labels = [lab for _, lab in trainval]
    train, val = train_test_split(trainval, test_size=val_frac, stratify=labels, random_state=seed)
    # sorted -> the csv is the same on every machine, not only the same set
    splits = {"train": sorted(train), "val": sorted(val), "test": test}

    for name, rows in splits.items():
        write_csv(rows, SPLIT_DIR / f"{name}.csv")
    return splits


def class_counts(splits, classes):
    """Images per breed in each split, saved to splits/class_counts.csv"""
    counts = {name: Counter(lab for _, lab in rows) for name, rows in splits.items()}
    with open(SPLIT_DIR / "class_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "breed", *counts.keys()])
        for i, breed in enumerate(classes):
            w.writerow([i, breed, *(counts[s][i] for s in counts)])
    return counts


def get_transforms():
    """Eval = the exact preprocessing of the pre-trained weights."""
    weights = MobileNet_V3_Small_Weights.DEFAULT
    eval_tf = weights.transforms()  # resize 256 -> center crop 224 -> ImageNet mean/std
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=eval_tf.mean, std=eval_tf.std),
    ])
    return train_tf, eval_tf


class PetDataset(Dataset):
    """Reads the images listed in one split csv"""

    def __init__(self, split, transform=None):
        self.items = read_csv(SPLIT_DIR / f"{split}.csv")
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_id, label = self.items[idx]
        img = Image.open(PET_DIR / "images" / f"{image_id}.jpg").convert("RGB")  # a few are RGBA/gray
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_loaders(batch_size=64, num_workers=4, seed=SEED, augment=True):
    """train / val / test DataLoaders. augment=False -> train set with eval transforms"""
    train_tf, eval_tf = get_transforms()
    g = torch.Generator().manual_seed(seed)  # fixed shuffling order
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return {
        "train": DataLoader(PetDataset("train", train_tf if augment else eval_tf),
                            shuffle=True, generator=g, **common),
        "val": DataLoader(PetDataset("val", eval_tf), shuffle=False, **common),
        "test": DataLoader(PetDataset("test", eval_tf), shuffle=False, **common),
    }


def get_classes():
    return OxfordIIITPet(RAW_DIR, split="test", target_types="category", download=False).classes


def main():
    trainval, test = download()
    classes = trainval.classes
    print(f"Official partitions: trainval {len(trainval)}, test {len(test)}, {len(classes)} breeds")

    splits = make_splits()
    counts = class_counts(splits, classes)
    print(f"Split (seed {SEED}): " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))

    print(f"\n{'breed':<28}{'train':>6}{'val':>6}{'test':>6}")
    for i, breed in enumerate(classes):
        print(f"{breed:<28}{counts['train'][i]:>6}{counts['val'][i]:>6}{counts['test'][i]:>6}")
    print(f"\nSplit files written to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
