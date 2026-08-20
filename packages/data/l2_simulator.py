"""Parametric Layer 2 fraud simulator.

Ground-truth-labelled fraud native to L2 transaction streams is not publicly
available at usable scale. This simulator injects the two target fraud archetypes
into synthetic directed transaction subgraphs with configurable base rates and
value distributions, so that heterogeneous (non-IID) client partitions can be
generated deterministically for the federated experiments.

Fraud archetypes:
  wash trade: a small closed set of addresses transacting in a directed cycle to
              fabricate volume. Topologically a tight cycle.
  flash loan: a single address opening and closing a large position across many
              counterparties within one block. Topologically a high-degree star.

Each generated block is a networkx.DiGraph with per-node attributes:
  value_in, value_out, degree_in, degree_out, is_fraud (0/1), fraud_type (str).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class SimConfig:
    n_blocks: int = 40
    nodes_per_block: int = 300
    edges_per_block: int = 600
    wash_rate: float = 0.15          # fraction of blocks that contain a wash cycle
    flash_rate: float = 0.15         # fraction of blocks that contain a flash-loan burst
    wash_cycle_len: int = 4
    flash_fanout: int = 25
    value_mean: float = 1.0
    value_std: float = 0.4
    seed: int = 7
    rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)


class L2FraudSimulator:
    """Generates a sequence of directed transaction graphs with injected fraud."""

    def __init__(self, config: SimConfig | None = None):
        self.cfg = config or SimConfig()

    def _base_block(self, block_idx: int) -> nx.DiGraph:
        cfg = self.cfg
        g = nx.gnm_random_graph(
            cfg.nodes_per_block, cfg.edges_per_block, seed=cfg.seed + block_idx, directed=True
        )
        g = nx.DiGraph(g)
        for _, _, data in g.edges(data=True):
            data["value"] = float(max(0.0, cfg.rng.normal(cfg.value_mean, cfg.value_std)))
        for n in g.nodes():
            g.nodes[n]["is_fraud"] = 0
            g.nodes[n]["fraud_type"] = "none"
        return g

    def _inject_wash_cycle(self, g: nx.DiGraph) -> None:
        cfg = self.cfg
        members = cfg.rng.choice(list(g.nodes()), size=cfg.wash_cycle_len, replace=False)
        for i in range(len(members)):
            src = int(members[i])
            dst = int(members[(i + 1) % len(members)])
            # Wash trades use inflated, repetitive values around a fixed level.
            g.add_edge(src, dst, value=float(cfg.value_mean * 5.0))
            g.nodes[src]["is_fraud"] = 1
            g.nodes[src]["fraud_type"] = "wash"

    def _inject_flash_burst(self, g: nx.DiGraph) -> None:
        cfg = self.cfg
        center = int(cfg.rng.choice(list(g.nodes())))
        counterparties = cfg.rng.choice(
            [n for n in g.nodes() if n != center], size=cfg.flash_fanout, replace=False
        )
        big = float(cfg.value_mean * 50.0)
        for cp in counterparties:
            g.add_edge(center, int(cp), value=big)   # borrow leg
            g.add_edge(int(cp), center, value=big)   # repay leg within same block
        g.nodes[center]["is_fraud"] = 1
        g.nodes[center]["fraud_type"] = "flash"

    def _finalise_features(self, g: nx.DiGraph) -> None:
        for n in g.nodes():
            in_edges = g.in_edges(n, data=True)
            out_edges = g.out_edges(n, data=True)
            g.nodes[n]["value_in"] = float(sum(d["value"] for _, _, d in in_edges))
            g.nodes[n]["value_out"] = float(sum(d["value"] for _, _, d in out_edges))
            g.nodes[n]["degree_in"] = g.in_degree(n)
            g.nodes[n]["degree_out"] = g.out_degree(n)

    def generate(self) -> list[nx.DiGraph]:
        cfg = self.cfg
        blocks: list[nx.DiGraph] = []
        for b in range(cfg.n_blocks):
            g = self._base_block(b)
            if cfg.rng.random() < cfg.wash_rate:
                self._inject_wash_cycle(g)
            if cfg.rng.random() < cfg.flash_rate:
                self._inject_flash_burst(g)
            self._finalise_features(g)
            blocks.append(g)
        return blocks
