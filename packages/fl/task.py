"""Shared training, evaluation, and parameter helpers.

These functions are the single code path used by both the centralised single-node
baseline (Phase 2) and the federated clients (Phase 3), so the federated result is
measured on identical logic. The FedProx proximal term is applied here so a client
can opt into it by passing the global reference parameters and a coefficient mu.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

from packages.models.gnn_lstm import GNNLSTMClassifier


# ----------------------------------------------------------------------------
# Parameter (de)serialisation for Flower
# ----------------------------------------------------------------------------

def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def build_model(in_dim: int, **kwargs) -> GNNLSTMClassifier:
    return GNNLSTMClassifier(in_dim=in_dim, **kwargs)


# ----------------------------------------------------------------------------
# Class-weighted loss (fraud is rare by construction)
# ----------------------------------------------------------------------------

def _class_weights(y: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    counts = torch.bincount(y[y >= 0], minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


# ----------------------------------------------------------------------------
# Training (with optional FedProx proximal term)
# ----------------------------------------------------------------------------

def train(
    model: GNNLSTMClassifier,
    snapshots: list,
    epochs: int = 3,
    lr: float = 1e-3,
    proximal_mu: float = 0.0,
    global_params: list[np.ndarray] | None = None,
    control_variate: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
    device: str = "cpu",
) -> dict:
    """Train the node classifier across a client's block snapshots.

    When proximal_mu > 0 and global_params are supplied, the FedProx proximal term
    (mu / 2) * || w - w_t ||^2 is added to the local loss to penalise drift away
    from the global model, stabilising training under non-IID data.

    When control_variate = (c_local, c_global) is supplied (SCAFFOLD), each step's
    gradient is corrected by (c_global - c_local) before the optimiser step, per
    the local update rule y <- y - eta_l * (g_k(y) - c_k + c). num_local_steps is
    returned so the caller can compute the SCAFFOLD control-variate update, which
    needs the count of local optimiser steps taken (K).
    """
    model.to(device).train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ref_tensors = None
    if proximal_mu > 0.0 and global_params is not None:
        ref_tensors = [torch.tensor(p, device=device) for p in global_params]

    scaffold_correction = None
    if control_variate is not None:
        c_local, c_global = control_variate
        c_local_t = [torch.tensor(c, device=device) for c in c_local]
        c_global_t = [torch.tensor(c, device=device) for c in c_global]
        scaffold_correction = [cg - cl for cl, cg in zip(
            _iter_ref(model, c_local_t), _iter_ref(model, c_global_t)
        )]

    last_loss = 0.0
    n_examples = 0
    n_steps = 0

    for _ in range(epochs):
        temporal_state = None
        for snap in snapshots:
            snap = snap.to(device)
            mask = getattr(snap, "train_mask", None)
            y = snap.y
            if mask is None:
                mask = y >= 0

            logits = model.forward_node(snap.x, snap.edge_index, temporal_state)
            if mask.sum() == 0:
                continue

            weights = _class_weights(y).to(device)
            loss = F.cross_entropy(logits[mask], y[mask], weight=weights)

            if ref_tensors is not None:
                prox = 0.0
                for p, ref in zip(model.parameters(), _iter_ref(model, ref_tensors)):
                    prox = prox + ((p - ref) ** 2).sum()
                loss = loss + (proximal_mu / 2.0) * prox

            optim.zero_grad()
            loss.backward()
            if scaffold_correction is not None:
                for p, correction in zip(model.parameters(), scaffold_correction):
                    if p.grad is not None:
                        p.grad.add_(correction)
            optim.step()
            n_steps += 1

            # Carry temporal state forward across blocks (detached to bound the graph).
            with torch.no_grad():
                temporal_state = model.forward_sequence([snap]).detach()

            last_loss = float(loss.detach().cpu())
            n_examples += int(mask.sum())

    return {"loss": last_loss, "num_examples": max(1, n_examples), "num_local_steps": max(1, n_steps)}


def _iter_ref(model: nn.Module, ref_tensors: list[torch.Tensor]):
    """Align the flat global reference tensors to model.parameters() order.

    state_dict order and parameters() order coincide for this model, so we map the
    trainable parameter slice of the reference list. Buffers (if any) are skipped.
    """
    param_names = [n for n, _ in model.named_parameters()]
    state_names = list(model.state_dict().keys())
    ref_by_name = {name: t for name, t in zip(state_names, ref_tensors)}
    for name in param_names:
        yield ref_by_name[name]


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: GNNLSTMClassifier, snapshots: list, device: str = "cpu") -> dict:
    model.to(device).eval()
    all_true, all_pred = [], []
    total_loss, n = 0.0, 0
    temporal_state = None

    for snap in snapshots:
        snap = snap.to(device)
        mask = getattr(snap, "train_mask", None)
        y = snap.y
        if mask is None:
            mask = y >= 0
        if mask.sum() == 0:
            continue

        logits = model.forward_node(snap.x, snap.edge_index, temporal_state)
        loss = F.cross_entropy(logits[mask], y[mask])
        total_loss += float(loss) * int(mask.sum())
        n += int(mask.sum())

        preds = logits[mask].argmax(dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(y[mask].cpu().numpy().tolist())
        temporal_state = model.forward_sequence([snap]).detach()

    if n == 0:
        return {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "num_examples": 0}

    return {
        "loss": total_loss / n,
        "precision": float(precision_score(all_true, all_pred, zero_division=0)),
        "recall": float(recall_score(all_true, all_pred, zero_division=0)),
        "f1": float(f1_score(all_true, all_pred, zero_division=0)),
        "num_examples": n,
    }
