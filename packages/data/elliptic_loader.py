"""Elliptic Bitcoin dataset loader.

The Elliptic dataset (Weber et al., 2019) is a large, realistically structured
transaction graph with licit, illicit, and unknown labels, already segmented into
49 discrete time steps. Those time steps map naturally onto the block-sequence
abstraction the LSTM consumes, so this loader exposes the data as an ordered list
of per-time-step frames.

Expected raw files (place under data/elliptic/):
  elliptic_txs_features.csv   txId, time_step, then 165 anonymised features
  elliptic_txs_classes.csv    txId, class in {1 illicit, 2 licit, unknown}
  elliptic_txs_edgelist.csv   txId1, txId2 directed edges

Labels are resolved to a binary illicit indicator: illicit -> 1, licit -> 0,
unknown -> -1 (masked out of the supervised loss).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Elliptic feature layout constants.
_LABEL_ILLICIT = "1"
_LABEL_LICIT = "2"
_NUM_LOCAL_FEATS = 94   # first 94 columns are local features
_TIME_STEP_COL = 1      # second column of the feature file is the time step


@dataclass
class TimeStepFrame:
    """One Elliptic time step: node features, id-to-index map, and binary labels."""

    time_step: int
    tx_ids: np.ndarray          # (N,) original transaction ids for this step
    features: np.ndarray        # (N, F) float feature matrix
    labels: np.ndarray          # (N,) in {0, 1, -1}; -1 means unknown/masked


class EllipticLoader:
    """Loads the Elliptic CSVs and yields ordered per-time-step frames plus edges."""

    def __init__(self, root: str | Path = "data/elliptic", use_local_feats_only: bool = False):
        self.root = Path(root)
        self.use_local_feats_only = use_local_feats_only
        self._features: pd.DataFrame | None = None
        self._classes: pd.DataFrame | None = None
        self._edges: pd.DataFrame | None = None

    def load(self) -> "EllipticLoader":
        feats = pd.read_csv(self.root / "elliptic_txs_features.csv", header=None)
        classes = pd.read_csv(self.root / "elliptic_txs_classes.csv")
        edges = pd.read_csv(self.root / "elliptic_txs_edgelist.csv")

        # Name the feature columns: col 0 is txId, col 1 is time_step, rest are features.
        n_feats = feats.shape[1] - 2
        feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(n_feats)]

        # Resolve class labels to a binary illicit indicator with -1 for unknown.
        classes = classes.rename(columns={classes.columns[0]: "txId", classes.columns[1]: "class"})
        label_map = {_LABEL_ILLICIT: 1, _LABEL_LICIT: 0}
        classes["label"] = classes["class"].astype(str).map(label_map).fillna(-1).astype(int)

        self._features = feats
        self._classes = classes.set_index("txId")["label"]
        self._edges = edges.rename(
            columns={edges.columns[0]: "src", edges.columns[1]: "dst"}
        )
        return self

    def time_steps(self) -> list[int]:
        assert self._features is not None, "call load() first"
        return sorted(self._features["time_step"].unique().tolist())

    def frame(self, time_step: int) -> TimeStepFrame:
        assert self._features is not None and self._classes is not None
        block = self._features[self._features["time_step"] == time_step].copy()

        feat_cols = [c for c in block.columns if c.startswith("f")]
        if self.use_local_feats_only:
            feat_cols = feat_cols[:_NUM_LOCAL_FEATS]

        tx_ids = block["txId"].to_numpy()
        features = block[feat_cols].to_numpy(dtype=np.float32)
        labels = (
            block["txId"].map(self._classes).fillna(-1).astype(int).to_numpy()
        )
        return TimeStepFrame(time_step=time_step, tx_ids=tx_ids, features=features, labels=labels)

    def frames(self) -> list[TimeStepFrame]:
        return [self.frame(ts) for ts in self.time_steps()]

    def edges(self) -> pd.DataFrame:
        assert self._edges is not None, "call load() first"
        return self._edges
