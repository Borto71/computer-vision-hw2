# data/

- `data.py`: downloads Oxford-IIIT Pet through `torchvision.datasets.OxfordIIITPet` into `raw/`, writes the stratified train/val split (80/20 of the official `trainval`, seed 42) and the per-breed counts, and defines the transforms, the `PetDataset` class and the DataLoaders. Run it once with `python data/data.py`.
- `splits/`: `train.csv`, `val.csv`, `test.csv` (`image_id,label`) and `class_counts.csv`. Committed, so the partition can be rebuilt exactly.
- `raw/`: the downloaded images. Not in the repository (git-ignored); the dataset is CC BY-SA 4.0 and is downloaded from the official source.
