"""Hybrid GraphSAGE-LSTM fraud classifier.

Fraud on a rollup has two signatures. The spatial signature lives in the
transaction graph topology within a block (wash trades form tight cycles,
flash-loan exploits form high-degree stars). The temporal signature lives in how
block-level aggregate state evolves across the sequence of blocks.

The model composes two standard components without modifying their internal
equations:

  1. A GraphSAGE spatial encoder (Hamilton et al., 2017) applied per block. It is
     chosen over a plain GCN because it aggregates over sampled neighbourhoods
     inductively and therefore generalises to addresses never seen in training,
     which is mandatory for a live L2 stream.

  2. An LSTM temporal encoder over the sequence of per-block embeddings, capturing
     the sequential dependency that a purely spatial model cannot.

The contribution is the composition (a purpose-built pipeline for the L2 fraud
task), not a new formula for either base component.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class SpatialEncoder(nn.Module):
    """Two-layer GraphSAGE producing per-node spatial embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h  # (num_nodes, out_dim)


class GNNLSTMClassifier(nn.Module):
    """Per-node fraud classifier fusing spatial and temporal context.

    Two operating modes are supported by the same weights:

      forward_node    node-level classification within a single block (used for the
                      Elliptic per-time-step supervised signal and for live scoring).
      forward_sequence  sequence-level fusion: encode a window of blocks spatially,
                        pool each block to a block embedding, run the LSTM over the
                        window, and expose the temporally informed representation.
    """

    def __init__(
        self,
        in_dim: int,
        gnn_hidden: int = 64,
        gnn_out: int = 64,
        lstm_hidden: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.spatial = SpatialEncoder(in_dim, gnn_hidden, gnn_out, dropout)
        self.lstm = nn.LSTM(input_size=gnn_out, hidden_size=lstm_hidden, batch_first=True)

        # Node head consumes the spatial embedding fused with the current temporal state.
        self.node_head = nn.Sequential(
            nn.Linear(gnn_out + lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )
        self.lstm_hidden = lstm_hidden

    def encode_block(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Return per-node spatial embeddings and the pooled block embedding."""
        node_emb = self.spatial(x, edge_index)
        batch = torch.zeros(node_emb.size(0), dtype=torch.long, device=node_emb.device)
        block_emb = global_mean_pool(node_emb, batch)  # (1, gnn_out)
        return node_emb, block_emb

    def forward_node(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classify nodes in a single block, optionally conditioned on temporal state.

        temporal_state is the LSTM hidden state carried from prior blocks, shape
        (lstm_hidden,). When None, a zero temporal context is used.
        """
        node_emb, _ = self.encode_block(x, edge_index)
        if temporal_state is None:
            temporal_state = torch.zeros(self.lstm_hidden, device=node_emb.device)
        temporal = temporal_state.unsqueeze(0).expand(node_emb.size(0), -1)
        fused = torch.cat([node_emb, temporal], dim=1)
        return self.node_head(fused)  # (num_nodes, num_classes) logits

    def forward_sequence(self, snapshots: list) -> torch.Tensor:
        """Run the LSTM over a window of blocks and return the final temporal state.

        snapshots is an ordered list of PyG Data objects. Returns the last LSTM
        hidden state, shape (lstm_hidden,), which can then condition node-level
        classification of the next incoming block.
        """
        block_embs = []
        for snap in snapshots:
            _, block_emb = self.encode_block(snap.x, snap.edge_index)
            block_embs.append(block_emb)
        seq = torch.stack(block_embs, dim=1)  # (1, T, gnn_out)
        out, (h_n, _) = self.lstm(seq)
        return h_n.squeeze(0).squeeze(0)  # (lstm_hidden,)
