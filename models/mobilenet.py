import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

NUM_CLASSES = 37
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT  # ImageNet-1K


def build_model(mode="full", num_classes=NUM_CLASSES):
    """Pre-trained MobileNetV3-Small with the last layer replaced by a 37-way one"""
    if mode not in ("full", "head", "features"):
        raise ValueError(f"unknown mode {mode}")

    model = mobilenet_v3_small(weights=WEIGHTS)

    # replace the final fully connected layer: 1000 ImageNet classes -> 37 breeds
    in_features = model.classifier[3].in_features  # 1024
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if mode == "head":
        # freeze the convolutional feature extractor, the classifier stays trainable
        for p in model.features.parameters():
            p.requires_grad = False
    elif mode == "features":
        # nothing is trained, the network is a fixed feature extractor
        for p in model.parameters():
            p.requires_grad = False

    return model


def set_train_mode(model, mode):
    """model.train() for training, but a frozen feature extractor must stay in eval():
    otherwise BatchNorm keeps updating its running mean/var even with requires_grad=False"""
    model.train()
    if mode == "head":
        model.features.eval()
    elif mode == "features":
        model.eval()


def count_params(model):
    """(trainable, total) number of parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def extract_features(model, loader, device):
    """576-d vectors after the global average pooling, before the classifier.
    Returns (features [N, 576], labels [N]) as numpy arrays."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model.avgpool(model.features(x)).flatten(1)  # same path as forward(), classifier skipped
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


if __name__ == "__main__":
    # quick check: output size and trainable parameters of each approach
    for mode in ("full", "head", "features"):
        m = build_model(mode)
        tr, tot = count_params(m)
        out = m.eval()(torch.zeros(1, 3, 224, 224))
        print(f"{mode:<9} output {tuple(out.shape)}  trainable {tr:,} / {tot:,}")
