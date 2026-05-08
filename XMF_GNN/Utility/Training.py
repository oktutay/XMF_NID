"""
Training and evaluation helpers for XMF-GNN.

Paper Section 4.4 hyperparameters (Table 3):
  - optimizer        : Adam
  - loss             : NLL  (cross-entropy on log-softmax output)
  - batch size       : 64
  - learning rate    : [10^-2, 10^-5]   -- ReduceLROnPlateau
  - epochs           : 100
  - hidden size      : 64
  - attn size        : 32
  - pooling          : Mean

ReduceLROnPlateau spec (paper Sec. 4.4): "decay is triggered when the
validation metric improves by less than 0.01 for five consecutive epochs",
which maps to ``patience=5, threshold=0.01`` in PyTorch's scheduler. The
``min_lr`` is 1e-5 to match the lower end of the lr range.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


def make_optimizer_and_scheduler(model, lr: float = 1e-2,
                                 patience: int = 5,
                                 threshold: float = 0.01,
                                 min_lr: float = 1e-5):
    """Adam + ReduceLROnPlateau matching paper Sec. 4.4."""
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=patience,
        threshold=threshold, min_lr=min_lr,
    )
    return optim, sched


# ---------------------------------------------------------------------------
# Train / eval loops -- the XMF-GNN forward signature is
# (x_dict, edge_index_dict, batch). No edge_attr_dict (paper Sec. 3.2).
# ---------------------------------------------------------------------------


def train(train_loader, model, args, device: str = "cuda",
          val_loader: Optional[object] = None, log_every: int = 1):
    """Train ``model`` for ``args['epochs']`` epochs.

    If ``val_loader`` is provided the scheduler steps on validation accuracy
    rather than train accuracy (paper Sec. 4.4 says scheduler watches the
    "validation metric").
    """
    optim, sched = make_optimizer_and_scheduler(
        model,
        lr=args.get("lr", 1e-2),
        patience=args.get("scheduler_patience", 5),
        threshold=args.get("scheduler_threshold", 0.01),
        min_lr=args.get("min_lr", 1e-5),
    )

    history = {"loss": [], "train_acc": [], "val_acc": [], "lr": []}

    for epoch in range(args["epochs"]):
        model.train()
        running_loss = 0.0
        n_graphs = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args['epochs']}"):
            batch.to(device)
            optim.zero_grad()
            preds = model(batch.x_dict, batch.edge_index_dict, batch)
            label = batch.y
            loss = model.loss(preds, label)
            loss.backward()
            optim.step()
            running_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs
        running_loss /= max(n_graphs, 1)

        train_acc = test(train_loader, model, device)
        val_acc = (test(val_loader, model, device)
                   if val_loader is not None else train_acc)
        sched.step(val_acc)
        cur_lr = optim.param_groups[0]["lr"]

        history["loss"].append(running_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        if (epoch + 1) % log_every == 0:
            print(
                f"Epoch {epoch + 1:3d}: "
                f"loss={running_loss:.4f}  "
                f"train_acc={train_acc:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"lr={cur_lr:.6f}"
            )

    return history


def test(loader, model, device: str = "cuda") -> float:
    """Return overall accuracy on ``loader``."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            preds = model(batch.x_dict, batch.edge_index_dict, batch).max(dim=1)[1]
            correct += preds.eq(batch.y).sum().item()
            total += batch.num_graphs
    return correct / max(total, 1)


def test_cm(loader, model, device: str = "cuda"):
    """Run inference and return ``(accuracy, predictions, labels)`` arrays.

    Also prints precision / recall / F1 (macro and weighted) plus the
    confusion matrix, mirroring paper Sec. 4.6.1 metric reporting.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch.to(device)
            preds = model(batch.x_dict, batch.edge_index_dict, batch).max(dim=1)[1]
            label = batch.y
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            correct += preds.eq(label).sum().item()
            total += batch.num_graphs

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels)
    return correct / max(total, 1), all_preds, all_labels


def calculate_metrics(y_pred, y_true) -> None:
    """Print a paper-Sec.-4.6 style metrics block."""
    print(f"\nConfusion matrix (rows = predicted, cols = true):\n{confusion_matrix(y_pred, y_true)}")
    print(f"Accuracy             : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (macro)    : {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall    (macro)    : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1        (macro)    : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Precision (weighted) : {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall    (weighted) : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1        (weighted) : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")


# ---------------------------------------------------------------------------
# Diagnostic: collect per-batch fusion attention weights -- used for
# paper Fig. 8 (flow attention vs epoch).
# ---------------------------------------------------------------------------


def collect_flow_attention(loader, model, device: str = "cuda") -> np.ndarray:
    """Run a forward pass over ``loader`` and return the average flow-side
    attention weight (alpha_f) over all graphs.

    Returns a single float: mean alpha_f across the loader. Use this once
    per epoch during training to reproduce Fig. 8(a)-(d).
    """
    model.eval()
    alphas = []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            _ = model(batch.x_dict, batch.edge_index_dict, batch)
            a = model.get_last_alpha()
            if a is None:
                continue
            # ``a`` shape can be (B, 2, 1) for ModalityFusion, (B, 2) for
            # SimpleAttn, (B, hidden) for Gated, (B, H, 2, 1) for MultiHead.
            if a.dim() == 3 and a.shape[1] == 2:
                alphas.append(a[:, 0, :].mean().item())
            elif a.dim() == 2 and a.shape[1] == 2:
                alphas.append(a[:, 0].mean().item())
            elif a.dim() == 4 and a.shape[2] == 2:
                alphas.append(a[:, :, 0, :].mean().item())
            else:
                alphas.append(a.mean().item())
    return float(np.mean(alphas)) if alphas else float("nan")
