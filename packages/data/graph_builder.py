"""Graph construction and non-IID partitioning.

Converts either the Elliptic frames or the L2 simulator output into a sequence of
torch_geometric.data.Data snapshots (one per block or time step), each carrying a
node feature tensor, an edge index, and a node-level label vector. Also provides a
non-IID partitioner that splits the node population across clients with skewed
fraud base rates, so the federated experiments run on genuinely heterogeneous data.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from .elliptic_loader import EllipticLoader, TimeStepFrame


# ----------------------------------------------------------------------------
# From the L2 simulator (networkx DiGraph blocks)
# ----------------------------------------------------------------------------

_L2_FEATURE_KEYS = ["value_in", "value_out", "degree_in", "degree_out"]


def block_to_pyg(g: nx.DiGraph) -> Data:
    """Convert one simulated block into a PyG Data snapshot."""
    nodes = list(g.nodes())
    index = {n: i for i, n in enumerate(nodes)}

    x = torch.tensor(
        [[float(g.nodes[n][k]) for k in _L2_FEATURE_KEYS] for n in nodes],
        dtype=torch.float32,
    )
    y = torch.tensor([int(g.nodes[n]["is_fraud"]) for n in nodes], dtype=torch.long)

    if g.number_of_edges() > 0:
        edge_index = torch.tensor(
            [[index[u], index[v]] for u, v in g.edges()], dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def simulator_to_snapshots(blocks: list[nx.DiGraph]) -> list[Data]:
    """Convert a full simulated block sequence into an ordered list of snapshots."""
    return [block_to_pyg(g) for g in blocks]


# ----------------------------------------------------------------------------
# From the Elliptic loader (per time-step frames)
# ----------------------------------------------------------------------------

def elliptic_to_snapshots(loader: EllipticLoader) -> list[Data]:
    """Convert Elliptic frames into per-time-step PyG snapshots.

    Edges are filtered to those whose endpoints both appear in the given time step,
    which respects the temporal segmentation of the dataset.
    """
    edges = loader.edges()
    snapshots: list[Data] = []
    for frame in loader.frames():
        snapshots.append(_frame_to_pyg(frame, edges))
    return snapshots


def _frame_to_pyg(frame: TimeStepFrame, edges) -> Data:
    id_to_idx = {int(tx): i for i, tx in enumerate(frame.tx_ids)}
    present = set(id_to_idx.keys())

    x = torch.tensor(frame.features, dtype=torch.float32)
    y = torch.tensor(frame.labels, dtype=torch.long)  # -1 marks unknown/masked

    mask = edges["src"].isin(present) & edges["dst"].isin(present)
    step_edges = edges[mask]
    if len(step_edges) > 0:
        edge_index = torch.tensor(
            [[id_to_idx[int(s)], id_to_idx[int(d)]] for s, d in
             zip(step_edges["src"], step_edges["dst"])],
            dtype=torch.long,
        ).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    labelled = (y >= 0)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = labelled
    return data


# ----------------------------------------------------------------------------
# Non-IID partitioning across clients
# ----------------------------------------------------------------------------

def partition_non_iid(
    snapshots: list[Data],
    num_clients: int,
    fraud_skew: float = 0.6,
    seed: int = 13,
) -> list[list[Data]]:
    """Split each snapshot's nodes across clients with skewed fraud base rates.

    Fraud-positive nodes are allocated to clients with a Dirichlet-controlled skew
    so that some clients see far more fraud than others, reproducing the non-IID
    condition characteristic of heterogeneous L2 nodes. Returns one snapshot
    sequence per client.
    """
    rng = np.random.default_rng(seed)
    client_streams: list[list[Data]] = [[] for _ in range(num_clients)]

    # Dirichlet concentration controls how uneven the fraud allocation is.
    alpha = np.full(num_clients, max(1e-3, 1.0 - fraud_skew))

    for snap in snapshots:
        n = snap.num_nodes
        y = snap.y.numpy()
        fraud_idx = np.where(y == 1)[0]
        clean_idx = np.where(y != 1)[0]

        # Clean nodes: roughly even split. Fraud nodes: skewed Dirichlet split.
        clean_assign = rng.integers(0, num_clients, size=len(clean_idx))
        fraud_props = rng.dirichlet(alpha)
        fraud_assign = rng.choice(num_clients, size=len(fraud_idx), p=fraud_props)

        for c in range(num_clients):
            keep = np.concatenate([
                clean_idx[clean_assign == c],
                fraud_idx[fraud_assign == c],
            ])
            if len(keep) == 0:
                continue
            client_streams[c].append(_subgraph(snap, np.sort(keep)))

    return client_streams


def _subgraph(data: Data, keep_idx: np.ndarray) -> Data:
    """Induce the node-subgraph on keep_idx, remapping the edge index."""
    keep = torch.as_tensor(keep_idx, dtype=torch.long)
    remap = -torch.ones(data.num_nodes, dtype=torch.long)
    remap[keep] = torch.arange(keep.numel())

    x = data.x[keep]
    y = data.y[keep]

    if data.edge_index.numel() > 0:
        ei = data.edge_index
        edge_keep = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
        edge_index = torch.stack([remap[ei[0][edge_keep]], remap[ei[1][edge_keep]]])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def client_fraud_rates(client_streams: list[list[Data]]) -> list[float]:
    """Report per-client positive-class ratio, to verify the partitions are non-IID."""
    rates = []
    for stream in client_streams:
        total = sum(int(s.num_nodes) for s in stream)
        pos = sum(int((s.y == 1).sum()) for s in stream)
        rates.append(pos / total if total else 0.0)
    return rates


# ----------------------------------------------------------------------------
# Stratified train/val split
# ----------------------------------------------------------------------------

def stratified_split(
    snapshots: list[Data], val_frac: float = 0.2, seed: int = 11
) -> tuple[list[Data], list[Data]]:
    """Split a snapshot list into train/val, holding out fraud-carrying blocks for both.

    A positional tail-slice biases which blocks land in validation: fraud events
    are sparse and seeded independently per block, so a fixed 80/20 split on
    unshuffled order can (and did, on this simulator's default config) leave
    validation with zero positive examples purely by chance, making
    precision/recall/f1 trivially 0 regardless of model quality. Splitting
    fraud-carrying and clean blocks separately (each shuffled, each guaranteed
    at least one block on both sides when available) ensures validation
    actually exercises the positive class. Shared by the FL client and the
    single-node baseline trainer so both are evaluated the same way.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(snapshots))
    has_fraud = np.array([int((s.y == 1).sum()) > 0 for s in snapshots])
    fraud_idx = rng.permutation(idx[has_fraud])
    clean_idx = rng.permutation(idx[~has_fraud])

    def _cut(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(arr) <= 1:
            return arr, np.array([], dtype=int)
        n_val = max(1, round(len(arr) * val_frac))
        return arr[n_val:], arr[:n_val]

    f_train, f_val = _cut(fraud_idx)
    c_train, c_val = _cut(clean_idx)
    train_idx = sorted(np.concatenate([f_train, c_train]))
    val_idx = sorted(np.concatenate([f_val, c_val]))
    return [snapshots[i] for i in train_idx], [snapshots[i] for i in val_idx]
