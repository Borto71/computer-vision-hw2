"""Training loop for the two PyTorch approaches (full fine-tuning and head training)."""

import time

import torch

from models.mobilenet import set_train_mode
from utils.metrics import compute_metrics, predict


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One pass over the training set. Returns (mean loss, accuracy)."""
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def fit(model, mode, loaders, device, lr, epochs, ckpt_path):
    """Train for `epochs`, keep the checkpoint with the best validation accuracy.
    Returns (history, train time in seconds). The best weights are loaded back into `model`."""
    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]  # head mode: only the classifier
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = -1.0
    start = time.time()

    for epoch in range(1, epochs + 1):
        set_train_mode(model, mode)  # frozen BatchNorm stays in eval for head mode
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        scheduler.step()

        y_true, y_prob, val_loss = predict(model, loaders["val"], device, criterion)
        val_acc = compute_metrics(y_true, y_prob)["accuracy"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"  epoch {epoch:2d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_acc:  # model selection on validation only
            best_acc = val_acc
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), ckpt_path)

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - start

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  selected epoch {history['best_epoch']} (val acc {best_acc:.4f}), "
          f"time {train_time:.1f}s")
    return history, train_time
