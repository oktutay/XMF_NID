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

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
    """Train ``model`` for up to ``args['epochs']`` epochs.

    Supports early stopping (``args['early_stopping_patience']``, default
    disabled) and saves the best-val checkpoint to
    ``args['best_checkpoint_path']`` whenever val_acc improves.

    If ``val_loader`` is provided the scheduler steps on validation accuracy
    rather than train accuracy (paper Sec. 4.4).
    """
    optim, sched = make_optimizer_and_scheduler(
        model,
        lr=args.get("lr", 1e-2),
        patience=args.get("scheduler_patience", 5),
        threshold=args.get("scheduler_threshold", 0.01),
        min_lr=args.get("min_lr", 1e-5),
    )

    history = {"loss": [], "train_acc": [], "val_acc": [], "lr": []}

    es_patience: Optional[int] = args.get("early_stopping_patience", None)
    best_val     = -1.0
    no_improve   = 0
    best_ckpt    = args.get("best_checkpoint_path", None)

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

        # Save best checkpoint whenever val_acc improves
        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            if best_ckpt:
                import os, torch as _torch
                os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
                _torch.save({"state_dict": model.state_dict(),
                             "epoch": epoch + 1,
                             "val_acc": val_acc,
                             "args": args}, best_ckpt)
        else:
            no_improve += 1

        if (epoch + 1) % log_every == 0:
            es_info = (f"  no_improve={no_improve}/{es_patience}"
                       if es_patience else "")
            print(
                f"Epoch {epoch + 1:3d}: "
                f"loss={running_loss:.4f}  "
                f"train_acc={train_acc:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"lr={cur_lr:.6f}"
                f"{es_info}"
            )

        # Early stopping
        if es_patience and no_improve >= es_patience:
            print(f"Early stopping at epoch {epoch + 1} "
                  f"(val_acc={val_acc:.4f}, best={best_val:.4f}, "
                  f"no improvement for {es_patience} epochs)")
            break

    return history


def _run_inference(loader, model, device: str = "cuda"
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model over ``loader`` and return (probs, preds, labels).

    Model output is log-softmax (NLLLoss) so we exponentiate.
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch)
            probs = out.exp()
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    if not all_probs:
        return (np.zeros((0, 0)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int))
    return (np.concatenate(all_probs, axis=0),
            np.concatenate(all_preds).ravel().astype(int),
            np.concatenate(all_labels).ravel().astype(int))


def test(loader, model, device: str = "cuda") -> float:
    """Return overall accuracy on ``loader``."""
    _, preds, labels = _run_inference(loader, model, device)
    return float((preds == labels).mean()) if len(labels) else 0.0


def test_cm(loader, model, device: str = "cuda", benign_label: int = 0,
            verbose: bool = True):
    """Run inference and return ``(accuracy, predictions, labels)`` arrays.

    Also prints per-class metrics + binary zero-day metrics (FAR / DR /
    AUC / TPR@FPR / EER) treating ``benign_label`` as the "negative" class
    and every other class as "attack".
    """
    probs, preds, labels = _run_inference(loader, model, device)
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    if verbose:
        calculate_metrics(preds, labels)
        zero_day_metrics(probs, labels, benign_label=benign_label, verbose=True)
    return acc, preds, labels


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
# Zero-day / anomaly detection metrics.
#
# Binary view: benign vs. attack. Attack-score = 1 - P(benign) from softmax.
# Reported metrics:
#   FAR   -- False Alarm Rate     = FP / (FP + TN)         (lower is better)
#   DR    -- Detection Rate       = TP / (TP + FN)         (higher is better)
#   MDR   -- Missed Detection     = 1 - DR
#   AUC-ROC, AUC-PR
#   TPR@FPR=1%, TPR@FPR=0.1%      (industry IDS operating points)
#   EER   -- Equal Error Rate     (FAR == MDR)
# ---------------------------------------------------------------------------


def _binary_attack_score(probs: np.ndarray, benign_label: int) -> np.ndarray:
    """Attack-score per sample = 1 - softmax probability of benign class."""
    if probs.size == 0:
        return np.zeros((0,))
    return 1.0 - probs[:, benign_label]


def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """Return TPR at the largest FPR not exceeding ``target_fpr``."""
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def _equal_error_rate(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """EER = point where FPR == 1 - TPR (i.e. FAR == miss rate)."""
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def zero_day_metrics(probs: np.ndarray, y_true: np.ndarray,
                     benign_label: int = 0,
                     decision_threshold: float = 0.5,
                     verbose: bool = True) -> Dict[str, float]:
    """Compute binary benign-vs-attack metrics suitable for zero-day reporting.

    Args:
        probs: (N, K) softmax probabilities from the classifier.
        y_true: (N,) integer class labels.
        benign_label: integer id of the benign class.
        decision_threshold: attack-score threshold for flagging an alarm.
        verbose: print a one-block summary.
    """
    if len(y_true) == 0:
        return {}
    y_attack = (y_true != benign_label).astype(int)  # 1 = attack, 0 = benign
    score = _binary_attack_score(probs, benign_label)
    pred = (score >= decision_threshold).astype(int)

    tn = int(((y_attack == 0) & (pred == 0)).sum())
    fp = int(((y_attack == 0) & (pred == 1)).sum())
    fn = int(((y_attack == 1) & (pred == 0)).sum())
    tp = int(((y_attack == 1) & (pred == 1)).sum())

    far = fp / max(fp + tn, 1)            # False Alarm Rate (= FPR)
    dr  = tp / max(tp + fn, 1)            # Detection Rate (= TPR / recall)
    mdr = 1.0 - dr
    prec_atk = tp / max(tp + fp, 1)
    f1_atk = (2 * prec_atk * dr / max(prec_atk + dr, 1e-12)) if (prec_atk + dr) else 0.0

    out = {
        "FAR": far, "DR": dr, "MDR": mdr,
        "Precision_attack": prec_atk, "F1_attack": f1_atk,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }

    # Score-based metrics require at least one positive AND one negative.
    if y_attack.min() == 0 and y_attack.max() == 1:
        fpr, tpr, _ = roc_curve(y_attack, score)
        out["AUC_ROC"]     = float(roc_auc_score(y_attack, score))
        out["AUC_PR"]      = float(average_precision_score(y_attack, score))
        out["TPR@FPR=1%"]  = _tpr_at_fpr(fpr, tpr, 0.01)
        out["TPR@FPR=0.1%"] = _tpr_at_fpr(fpr, tpr, 0.001)
        out["EER"]         = _equal_error_rate(fpr, tpr)

    if verbose:
        print("\nZero-day binary metrics (benign vs attack)")
        print(f"  threshold        : {decision_threshold:.2f}")
        print(f"  TP/FP/TN/FN      : {tp} / {fp} / {tn} / {fn}")
        print(f"  FAR (FPR)        : {far:.4f}")
        print(f"  DR  (TPR/recall) : {dr:.4f}")
        print(f"  MDR              : {mdr:.4f}")
        print(f"  Precision_attack : {prec_atk:.4f}")
        print(f"  F1_attack        : {f1_atk:.4f}")
        if "AUC_ROC" in out:
            print(f"  AUC-ROC          : {out['AUC_ROC']:.4f}")
            print(f"  AUC-PR           : {out['AUC_PR']:.4f}")
            print(f"  TPR @ FPR=1%     : {out['TPR@FPR=1%']:.4f}")
            print(f"  TPR @ FPR=0.1%   : {out['TPR@FPR=0.1%']:.4f}")
            print(f"  EER              : {out['EER']:.4f}")
    return out


def collect_dataset_labels(dataset, cache_path: Optional[str] = None) -> np.ndarray:
    """Iterate dataset once, return ``(N,)`` int64 labels.

    Optionally cache to ``cache_path`` (``.npy``) for re-use.
    """
    import os
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached) == len(dataset):
            return cached
    labels = np.empty(len(dataset), dtype=np.int64)
    for i in tqdm(range(len(dataset)), desc="collecting labels"):
        labels[i] = int(dataset.get(i).y.item())
    if cache_path:
        np.save(cache_path, labels)
    return labels


def make_loao_subsets(train_set, test_set, held_out_label_id: int,
                      cache_dir: Optional[str] = None):
    """Split (train_set, test_set) for leave-one-attack-out evaluation.

    Returns:
        (train_known, test_known, test_unseen) -- ``torch.utils.data.Subset``
        objects compatible with ``torch_geometric.loader.DataLoader``.

    The model still outputs ``K`` classes; the held-out label simply never
    appears in training. Detection on unseen samples is then "model
    predicted anything != benign".
    """
    import os
    from torch.utils.data import Subset

    train_cache = os.path.join(cache_dir, "train_labels.npy") if cache_dir else None
    test_cache  = os.path.join(cache_dir, "test_labels.npy")  if cache_dir else None
    train_labels = collect_dataset_labels(train_set, cache_path=train_cache)
    test_labels  = collect_dataset_labels(test_set,  cache_path=test_cache)

    train_idx       = np.where(train_labels != held_out_label_id)[0].tolist()
    test_known_idx  = np.where(test_labels  != held_out_label_id)[0].tolist()
    test_unseen_idx = np.where(test_labels  == held_out_label_id)[0].tolist()
    return (Subset(train_set, train_idx),
            Subset(test_set,  test_known_idx),
            Subset(test_set,  test_unseen_idx))


def loao_evaluate(model, known_loader, unseen_loader,
                  benign_label: int = 0,
                  decision_threshold: float = 0.5,
                  device: str = "cuda") -> Dict[str, float]:
    """Evaluate a model trained on K-1 classes against a held-out attack class.

    Args:
        known_loader: test loader containing only classes seen during training.
        unseen_loader: test loader containing ONLY the held-out attack class.
        benign_label: id of the benign class in the K-1 training label space.

    For the unseen class the classifier cannot output the correct label
    (it wasn't in training). "Detection" is therefore: the model predicts
    anything other than benign.
    """
    probs_k, preds_k, labels_k = _run_inference(known_loader, model, device)
    probs_u, preds_u, _        = _run_inference(unseen_loader, model, device)

    # Known-class side: ordinary multi-class accuracy + binary zero-day view.
    acc_known = float((preds_k == labels_k).mean()) if len(labels_k) else float("nan")
    far_metrics = zero_day_metrics(probs_k, labels_k, benign_label=benign_label,
                                   decision_threshold=decision_threshold, verbose=False)
    far = far_metrics.get("FAR", float("nan"))

    # Unseen side: how often is the held-out attack flagged as non-benign?
    if len(preds_u):
        score_u = _binary_attack_score(probs_u, benign_label)
        dr_unseen = float((preds_u != benign_label).mean())
        dr_unseen_thresh = float((score_u >= decision_threshold).mean())
        score_mean = float(score_u.mean())
    else:
        dr_unseen = dr_unseen_thresh = score_mean = float("nan")

    out = {
        "accuracy_known": acc_known,
        "FAR_on_benign": far,
        "DR_unseen_argmax": dr_unseen,
        "DR_unseen_threshold": dr_unseen_thresh,
        "mean_attack_score_unseen": score_mean,
        "n_known_samples": int(len(labels_k)),
        "n_unseen_samples": int(len(preds_u)),
    }
    print("\nLeave-one-attack-out evaluation")
    print(f"  known classes : {len(labels_k)} samples, accuracy={acc_known:.4f}")
    print(f"  FAR (benign)  : {far:.4f}")
    print(f"  unseen class  : {len(preds_u)} samples")
    print(f"    DR (argmax != benign)      : {dr_unseen:.4f}")
    print(f"    DR (attack_score >= {decision_threshold:.2f}) : {dr_unseen_thresh:.4f}")
    print(f"    mean attack_score          : {score_mean:.4f}")
    return out


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
