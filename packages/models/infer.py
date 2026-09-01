"""Ad-hoc transaction scoring against a trained checkpoint.

Reuses the exact 4-feature schema the L2 simulator trains on (value_in,
value_out, degree_in, degree_out -- see packages/data/l2_simulator.py
_finalise_features), so a hand-built mini transaction graph can be scored the
same way a simulated block is: build the node set from the submitted edges,
derive per-node features from those edges, and run the same forward_node path
task.py uses for training and evaluation. Elliptic-trained checkpoints (165
anonymised features) can't be scored this way -- there is no meaningful way for
a user to hand-type 165 opaque statistics -- so only simulator-schema
checkpoints (in_dim == 4) are accepted here.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from packages.fl import task
from packages.models.reasons import explain

_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
_SIM_IN_DIM = 4

# Loaded models are cheap to keep resident and the checkpoint set doesn't
# change while the backend process is running, so cache by checkpoint name.
_model_cache: dict[str, tuple[torch.nn.Module, dict]] = {}


class ScoringError(Exception):
    """Raised for any user-correctable problem (bad checkpoint, empty graph)."""


def available_checkpoints() -> list[dict]:
    """List checkpoints on disk with enough metadata for a model picker."""
    out = []
    for path in sorted(_CHECKPOINT_DIR.glob("*.pt")):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        out.append({
            "name": path.stem,
            "in_dim": checkpoint.get("in_dim"),
            "scorable": checkpoint.get("in_dim") == _SIM_IN_DIM,
            "source_or_strategy": checkpoint.get("source", checkpoint.get("strategy", "unknown")),
            "metrics": checkpoint.get("metrics", {}),
        })
    return out


def _load(name: str) -> tuple[torch.nn.Module, dict]:
    if name in _model_cache:
        return _model_cache[name]

    path = _CHECKPOINT_DIR / f"{name}.pt"
    if not path.exists():
        raise ScoringError(f"no checkpoint named '{name}' in {_CHECKPOINT_DIR}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    in_dim = checkpoint.get("in_dim")
    if in_dim != _SIM_IN_DIM:
        raise ScoringError(
            f"checkpoint '{name}' was trained on {in_dim}-dim features (Elliptic); "
            f"the transaction tester only supports simulator-schema models "
            f"(in_dim={_SIM_IN_DIM}: value_in, value_out, degree_in, degree_out)"
        )

    model = task.build_model(in_dim=in_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _model_cache[name] = (model, checkpoint)
    return _model_cache[name]


def _build_graph(edges: list[dict], target: str) -> tuple[Data, int, dict[str, dict]]:
    nodes: list[str] = []
    seen: set[str] = set()
    for e in edges:
        for n in (str(e["src"]), str(e["dst"])):
            if n not in seen:
                seen.add(n)
                nodes.append(n)
    if target not in seen:
        nodes.append(target)
    index = {n: i for i, n in enumerate(nodes)}

    stats = {n: {"value_in": 0.0, "value_out": 0.0, "degree_in": 0, "degree_out": 0} for n in nodes}
    for e in edges:
        src, dst, value = str(e["src"]), str(e["dst"]), float(e["value"])
        stats[src]["value_out"] += value
        stats[src]["degree_out"] += 1
        stats[dst]["value_in"] += value
        stats[dst]["degree_in"] += 1

    x = torch.tensor(
        [[stats[n]["value_in"], stats[n]["value_out"], stats[n]["degree_in"], stats[n]["degree_out"]]
         for n in nodes],
        dtype=torch.float32,
    )
    if edges:
        edge_index = torch.tensor(
            [[index[str(e["src"])], index[str(e["dst"])]] for e in edges], dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index), index[target], stats


@torch.no_grad()
def score_transaction(checkpoint_name: str, edges_in: list[dict], target: str) -> dict:
    if not edges_in:
        raise ScoringError("at least one edge is required to build a scoring context")

    model, checkpoint = _load(checkpoint_name)
    data, target_idx, stats = _build_graph(edges_in, target)

    logits = model.forward_node(data.x, data.edge_index, temporal_state=None)
    probs = torch.softmax(logits, dim=1)
    fraud_prob = float(probs[target_idx, 1])

    reasons = explain(edges=edges_in, target=target, node_stats=stats[target])

    return {
        "checkpoint": checkpoint_name,
        "target": target,
        "probability": round(fraud_prob, 4),
        "is_fraud": fraud_prob >= 0.5,
        "features": stats[target],
        "reasons": reasons,
        "model_metrics": checkpoint.get("metrics", {}),
    }
