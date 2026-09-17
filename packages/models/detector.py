"""Fraud detection sweep over batches of L2 transactions.

The detector isn't wired into a live rollup node, so transactions reach it one
of two ways:

  simulated  Fresh blocks from packages/data/l2_simulator.py -- the same
             generator the federated clients train on -- with ground truth
             attached, so every scan can be scored for accuracy.
  upload     A CSV or JSON file of transfers supplied by the user. Ground truth
             is optional: include an is_fraud column and the scan is scored the
             same way; leave it out and the scan just reports what it flagged.

Every address in every block is scored by one of two detectors, selected by
FRAUD_DETECTION_MODE:

  model      The trained checkpoint, run over the whole block through the same
             forward_node path task.py trains and evaluates with (rolling
             temporal window included), so the numbers are comparable to a
             training run's.
  simulated  Verdicts synthesised from ground truth without loading a model,
             so the screen works before anything has been trained. Needs
             labels, so it can't check an unlabelled upload. Deliberately
             imperfect: it misses some fraud and raises the odd false alarm.

Both sources and both detectors produce the identical payload shape.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from packages.data.graph_builder import block_to_pyg
from packages.data.l2_simulator import L2FraudSimulator, SimConfig, finalise_features
from packages.models.infer import ScoringError, available_checkpoints, load_model
from packages.models.reasons import explain

MODE_MODEL = "model"
MODE_SIMULATED = "simulated"
MODES = (MODE_MODEL, MODE_SIMULATED)

FALLBACK_MODE = MODE_SIMULATED
FALLBACK_CHECKPOINT = "federated_fedprox"

DEFAULT_BLOCKS = 12
DEFAULT_THRESHOLD = 0.5
# Deliberately not seed 7: that's the stream packages/fl/client.py partitions
# into client shards and train_baseline.py trains on, so scanning it would score
# a checkpoint against the very blocks it learned from.
DEFAULT_SEED = 23
TRAINING_SEED = 7

MAX_UPLOAD_ROWS = 50_000

_TEMPORAL_WINDOW = 3        # matches task.train / task.evaluate
_MAX_CONTEXT_EDGES = 80     # bounds explain()'s cycle DFS on dense blocks
_MAX_DETAILED = 50          # per-scan cap on detections we build full detail for
_MAX_MISSED = 20            # per-scan cap on missed-fraud detail

# The false-alarm rate looks implausibly low until you account for the base
# rate: the simulator marks well under 1% of addresses fraudulent, so even a 2%
# chance of crying wolf buries every true positive under ~8x as many false ones.
# These land the synthetic detector near precision 0.8 / recall 0.85.
_SIM_MISS_RATE = 0.15
_SIM_FALSE_ALARM_RATE = 0.0007

_COLUMN_ALIASES = {
    "src": ("from", "src", "sender", "source"),
    "dst": ("to", "dst", "receiver", "recipient", "destination"),
    "value": ("value", "amount"),
    "block": ("block", "batch", "block_number"),
    "is_fraud": ("is_fraud", "fraud", "label"),
    "fraud_type": ("fraud_type", "type"),
}


def configured_mode() -> str:
    """The mode FRAUD_DETECTION_MODE asks for, read at call time.

    Not a module-level constant: backend/main.py loads .env at import, and a
    constant evaluated here would freeze whatever the environment held when
    this module happened to be imported.
    """
    return os.getenv("FRAUD_DETECTION_MODE", FALLBACK_MODE).strip().lower()


def configured_checkpoint() -> str:
    return os.getenv("FRAUD_CHECKPOINT", FALLBACK_CHECKPOINT).strip()


def default_checkpoint() -> str | None:
    """The configured checkpoint if it's scorable, else any scorable one on disk."""
    scorable = [c["name"] for c in available_checkpoints() if c["scorable"]]
    configured = configured_checkpoint()
    if configured in scorable:
        return configured
    return scorable[0] if scorable else None


def config() -> dict:
    checkpoint = default_checkpoint()
    return {
        "mode": configured_mode(),
        "modes": list(MODES),
        "checkpoint": checkpoint,
        "checkpoint_available": checkpoint is not None,
        "default_blocks": DEFAULT_BLOCKS,
        "default_threshold": DEFAULT_THRESHOLD,
        "default_seed": DEFAULT_SEED,
        "training_seed": TRAINING_SEED,
        "max_upload_rows": MAX_UPLOAD_ROWS,
    }


def model_info(checkpoint: str | None = None) -> dict:
    """What a checkpoint is and the scores recorded when it was trained."""
    name = checkpoint or default_checkpoint()
    if not name:
        return {"available": False}
    _, ck = load_model(name)
    federated = "strategy" in ck
    return {
        "available": True,
        "name": name,
        "kind": "federated" if federated else "centralised",
        "strategy": ck.get("strategy"),
        "rounds": ck.get("rounds"),
        "clients": ck.get("min_clients"),
        "trained_on": ck.get("source", "simulator"),
        "training_seed": TRAINING_SEED,
        "metrics": ck.get("metrics", {}),
    }


# ----------------------------------------------------------------------------
# Building blocks from uploaded files
# ----------------------------------------------------------------------------

def _pick(row: dict, field: str):
    for alias in _COLUMN_ALIASES[field]:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "fraud")


def parse_transactions(content: str, fmt: str) -> tuple[list[dict], bool]:
    """Parse an uploaded CSV/JSON file into transfer rows.

    Returns (rows, labelled). Error messages are written for the person who
    prepared the file, not for a developer, since that's who will read them.
    """
    fmt = (fmt or "").strip().lower()
    if fmt == "json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ScoringError(
                f"This file isn't valid JSON (problem on line {exc.lineno}: {exc.msg})."
            )
        if isinstance(data, dict) and isinstance(data.get("transactions"), list):
            data = data["transactions"]
        if not isinstance(data, list):
            raise ScoringError(
                "A JSON file should be a list of transactions, e.g. "
                '[{"from": "A", "to": "B", "value": 5}].'
            )
        raw = data
    elif fmt == "csv":
        raw = list(csv.DictReader(io.StringIO(content.lstrip("﻿"))))
    else:
        raise ScoringError("Please upload a .csv or .json file.")

    if not raw:
        raise ScoringError("The file has no transactions in it.")
    if len(raw) > MAX_UPLOAD_ROWS:
        raise ScoringError(
            f"The file has {len(raw):,} transactions; the limit is {MAX_UPLOAD_ROWS:,}. "
            "Split it into smaller files."
        )

    rows: list[dict] = []
    problems: list[str] = []
    labelled = False
    for number, entry in enumerate(raw, start=2 if fmt == "csv" else 1):
        if len(problems) >= 5:
            break
        if not isinstance(entry, dict):
            problems.append(f"item {number} isn't a transaction")
            continue
        entry = {str(k).strip().lower(): v for k, v in entry.items() if k is not None}

        src, dst, value = _pick(entry, "src"), _pick(entry, "dst"), _pick(entry, "value")
        where = f"line {number}" if fmt == "csv" else f"transaction {number}"
        if src is None or dst is None:
            problems.append(f"{where} is missing a 'from' or 'to' wallet")
            continue
        try:
            amount = float(value)
        except (TypeError, ValueError):
            problems.append(f"{where} has an amount that isn't a number ({value!r})")
            continue
        if not np.isfinite(amount) or amount < 0:
            problems.append(f"{where} has a negative or invalid amount")
            continue

        label = _pick(entry, "is_fraud")
        if any(alias in entry for alias in _COLUMN_ALIASES["is_fraud"]):
            labelled = True
        is_fraud = label is not None and _truthy(label)
        fraud_type = str(_pick(entry, "fraud_type") or "").strip().lower()

        rows.append({
            "src": str(src).strip(),
            "dst": str(dst).strip(),
            "value": amount,
            "block": str(_pick(entry, "block") or "1").strip(),
            "is_fraud": is_fraud,
            "fraud_type": (fraud_type or "unknown") if is_fraud else "none",
        })

    if problems:
        more = " (showing the first 5)" if len(problems) >= 5 else ""
        raise ScoringError("Some rows couldn't be read" + more + ": " + "; ".join(problems) + ".")
    return rows, labelled


def _graphs_from_rows(rows: list[dict]) -> tuple[list[str], list[nx.DiGraph]]:
    """Group transfers into one graph per block, with the simulator's features.

    Repeated transfers between the same pair in one block are summed into a
    single edge: the model was trained on DiGraph blocks, where degree counts
    distinct counterparties, so a per-transfer multigraph would inflate degrees
    beyond anything it has seen. Labels mark the sending wallet, matching how
    the simulator marks the addresses that drive a wash cycle or flash burst.
    """
    blocks: dict[str, nx.DiGraph] = {}
    for r in rows:
        g = blocks.setdefault(r["block"], nx.DiGraph())
        for n in (r["src"], r["dst"]):
            if n not in g:
                g.add_node(n, is_fraud=0, fraud_type="none", address=n)
        if g.has_edge(r["src"], r["dst"]):
            g[r["src"]][r["dst"]]["value"] += r["value"]
        else:
            g.add_edge(r["src"], r["dst"], value=r["value"])
        if r["is_fraud"]:
            g.nodes[r["src"]]["is_fraud"] = 1
            g.nodes[r["src"]]["fraud_type"] = r["fraud_type"]

    for g in blocks.values():
        finalise_features(g)
    return list(blocks.keys()), list(blocks.values())


# ----------------------------------------------------------------------------
# Scoring core, shared by both sources
# ----------------------------------------------------------------------------

def _pseudo_address(seed: int, block_idx: int, node) -> str:
    return "0x" + hashlib.sha256(f"addr:{seed}:{block_idx}:{node}".encode()).hexdigest()[:40]


def _pseudo_tx_hash(seed: int, block_idx: int, node) -> str:
    return "0x" + hashlib.sha256(f"tx:{seed}:{block_idx}:{node}".encode()).hexdigest()


@torch.no_grad()
def _model_probabilities(model, window: list[Data], snapshot: Data) -> list[float]:
    """Fraud probability for every node in the block, in block_to_pyg node order."""
    temporal_state = model.forward_sequence(window)
    logits = model.forward_node(snapshot.x, snapshot.edge_index, temporal_state)
    return torch.softmax(logits, dim=1)[:, 1].tolist()


def _simulated_probability(rng: np.random.Generator, is_fraud: int) -> float:
    if is_fraud:
        if rng.random() < _SIM_MISS_RATE:
            return float(rng.uniform(0.15, 0.45))
        return float(rng.uniform(0.72, 0.98))
    if rng.random() < _SIM_FALSE_ALARM_RATE:
        return float(rng.uniform(0.55, 0.80))
    return float(rng.uniform(0.0, 0.12))


def _context_edges(g: nx.DiGraph, node) -> list[dict]:
    """The 2-hop induced neighbourhood around `node` as an edge list.

    Two hops is the minimum that can contain a whole wash cycle (the simulator's
    default cycle is 4 addresses, all within two hops of any member), which is
    what reasons.explain needs to trace one. The cap keeps that trace cheap on a
    flash-loan centre, whose 2-hop neighbourhood spans hundreds of edges: the
    node's own edges are always kept, the rest highest-value first, since both
    archetypes transact well above the block average.
    """
    one_hop = {node} | set(g.successors(node)) | set(g.predecessors(node))
    two_hop = set(one_hop)
    for n in one_hop:
        two_hop |= set(g.successors(n)) | set(g.predecessors(n))

    incident, other = [], []
    for u, v, data in g.subgraph(two_hop).edges(data=True):
        edge = {"src": str(u), "dst": str(v), "value": round(float(data["value"]), 4)}
        (incident if node in (u, v) else other).append(edge)

    other.sort(key=lambda e: e["value"], reverse=True)
    return (incident + other)[:_MAX_CONTEXT_EDGES]


def _counterparties(g: nx.DiGraph, node) -> list[dict]:
    parties: dict = {}

    def touch(other, direction: str, value: float):
        entry = parties.setdefault(other, {
            "node": str(other),
            "address": g.nodes[other]["address"],
            "sent_to_target": 0.0,
            "received_from_target": 0.0,
            "edges": 0,
        })
        entry[direction] += value
        entry["edges"] += 1

    for src, _, data in g.in_edges(node, data=True):
        touch(src, "sent_to_target", float(data["value"]))
    for _, dst, data in g.out_edges(node, data=True):
        touch(dst, "received_from_target", float(data["value"]))

    out = []
    for entry in parties.values():
        both = entry["sent_to_target"] > 0 and entry["received_from_target"] > 0
        out.append({
            **entry,
            "sent_to_target": round(entry["sent_to_target"], 4),
            "received_from_target": round(entry["received_from_target"], 4),
            # Both legs with the same wallet inside one block is the flash-loan
            # borrow/repay signature reasons.py keys on.
            "direction": "both" if both else ("in" if entry["sent_to_target"] > 0 else "out"),
        })
    out.sort(key=lambda p: p["sent_to_target"] + p["received_from_target"], reverse=True)
    return out


def _detection(g, node, probability, block_idx, block_label, flagged, labelled, tx_hash):
    attrs = g.nodes[node]
    features = {
        "value_in": round(float(attrs["value_in"]), 4),
        "value_out": round(float(attrs["value_out"]), 4),
        "degree_in": int(attrs["degree_in"]),
        "degree_out": int(attrs["degree_out"]),
    }
    is_fraud = bool(attrs["is_fraud"])
    context = _context_edges(g, node)

    if not labelled:
        verdict = "flagged" if flagged else "not_flagged"
    elif flagged:
        verdict = "true_positive" if is_fraud else "false_positive"
    else:
        verdict = "false_negative" if is_fraud else "true_negative"

    return {
        "id": f"{block_idx}:{node}",
        "block": block_label,
        "node": str(node),
        "address": attrs["address"],
        "tx_hash": tx_hash,
        "probability": round(probability, 4),
        "flagged": flagged,
        "verdict": verdict,
        "features": features,
        "reasons": explain(edges=context, target=str(node), node_stats=features),
        "counterparties": _counterparties(g, node),
        "context_edges": context,
        "ground_truth": (
            {"is_fraud": is_fraud, "fraud_type": attrs["fraud_type"]} if labelled else None
        ),
    }


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarm_rate": round(fp / (fp + tn), 6) if (fp + tn) else 0.0,
    }


def _resolve_detector(mode: str | None, checkpoint: str | None, labelled: bool):
    mode = (mode or configured_mode()).strip().lower()
    if mode not in MODES:
        raise ScoringError(f"unknown mode '{mode}' (expected one of: {', '.join(MODES)})")

    if mode == MODE_SIMULATED:
        if not labelled:
            raise ScoringError(
                "The practice checker can only check a file that already says which "
                "wallets are fraudulent (an is_fraud column). Choose “The trained AI "
                "model” under “Who checks the transactions” to check this file."
            )
        return mode, None, None

    checkpoint = checkpoint or default_checkpoint()
    if not checkpoint:
        raise ScoringError(
            "No trained AI model was found. Train one first (see the README), "
            "or choose “Practice checker” under “Who checks the transactions”."
        )
    model, _ = load_model(checkpoint)
    return mode, checkpoint, model


def _scan_graphs(
    graphs: list[nx.DiGraph],
    block_labels: list,
    *,
    mode: str,
    model,
    threshold: float,
    rng: np.random.Generator,
    labelled: bool,
    tx_hash_for,
) -> dict:
    tp = fp = fn = tn = 0
    scanned = 0
    flagged_refs: list[tuple[float, int, str]] = []   # (probability, block_idx, node key)
    missed_refs: list[tuple[float, int, str]] = []
    by_type: dict[str, dict] = {}
    window: list[Data] = []
    node_lookup: list[dict] = []

    for block_idx, g in enumerate(graphs):
        nodes = list(g.nodes())
        node_lookup.append({str(n): n for n in nodes})
        labels = [int(g.nodes[n]["is_fraud"]) for n in nodes]

        if mode == MODE_MODEL:
            snapshot = block_to_pyg(g)
            window = (window + [snapshot])[-_TEMPORAL_WINDOW:]
            probabilities = _model_probabilities(model, window, snapshot)
        else:
            probabilities = [_simulated_probability(rng, y) for y in labels]

        for node, label, probability in zip(nodes, labels, probabilities):
            scanned += 1
            flagged = probability >= threshold
            if flagged:
                flagged_refs.append((probability, block_idx, str(node)))
            if not labelled:
                continue

            if label:
                kind = by_type.setdefault(g.nodes[node]["fraud_type"],
                                          {"total": 0, "caught": 0, "probability_sum": 0.0})
                kind["total"] += 1
                kind["probability_sum"] += probability
                if flagged:
                    kind["caught"] += 1
                    tp += 1
                else:
                    fn += 1
                    missed_refs.append((probability, block_idx, str(node)))
            elif flagged:
                fp += 1
            else:
                tn += 1

    # Detail extraction walks each address's neighbourhood, so it's bounded to
    # the strongest verdicts; the counts cover the full sweep regardless.
    flagged_refs.sort(reverse=True)
    missed_refs.sort(reverse=True)

    def build(refs, cap, flagged):
        out = []
        for probability, block_idx, key in refs[:cap]:
            node = node_lookup[block_idx][key]
            out.append(_detection(
                graphs[block_idx], node, probability, block_idx, block_labels[block_idx],
                flagged, labelled, tx_hash_for(block_idx, node),
            ))
        return out

    summary = {
        "blocks": len(graphs),
        "scanned": scanned,
        "flagged": len(flagged_refs),
        "detailed": min(len(flagged_refs), _MAX_DETAILED),
        "labelled": labelled,
    }
    if labelled:
        summary.update({
            "fraud_in_stream": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            **_metrics(tp, fp, fn, tn),
            "by_type": {
                kind: {
                    "total": v["total"],
                    "caught": v["caught"],
                    "catch_rate": round(v["caught"] / v["total"], 4),
                    "mean_probability": round(v["probability_sum"] / v["total"], 6),
                }
                for kind, v in sorted(by_type.items())
            },
        })

    return {
        "detections": build(flagged_refs, _MAX_DETAILED, True),
        "missed": build(missed_refs, _MAX_MISSED, False) if labelled else [],
        "summary": summary,
    }


def scan(
    blocks: int = DEFAULT_BLOCKS,
    threshold: float = DEFAULT_THRESHOLD,
    mode: str | None = None,
    checkpoint: str | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Sweep a freshly simulated block stream."""
    mode, checkpoint, model = _resolve_detector(mode, checkpoint, labelled=True)

    graphs = L2FraudSimulator(SimConfig(seed=seed, n_blocks=blocks)).generate()
    for block_idx, g in enumerate(graphs):
        for n in g.nodes():
            g.nodes[n]["address"] = _pseudo_address(seed, block_idx, n)

    result = _scan_graphs(
        graphs, list(range(len(graphs))),
        mode=mode, model=model, threshold=threshold,
        rng=np.random.default_rng(seed), labelled=True,
        tx_hash_for=lambda block_idx, node: _pseudo_tx_hash(seed, block_idx, node),
    )
    return {"source": "simulated", "mode": mode, "checkpoint": checkpoint,
            "threshold": threshold, "seed": seed, **result}


def scan_upload(
    content: str,
    fmt: str,
    threshold: float = DEFAULT_THRESHOLD,
    mode: str | None = None,
    checkpoint: str | None = None,
) -> dict:
    """Sweep transactions from an uploaded CSV/JSON file."""
    rows, labelled = parse_transactions(content, fmt)
    mode, checkpoint, model = _resolve_detector(mode, checkpoint, labelled)
    block_labels, graphs = _graphs_from_rows(rows)

    result = _scan_graphs(
        graphs, block_labels,
        mode=mode, model=model, threshold=threshold,
        rng=np.random.default_rng(DEFAULT_SEED), labelled=labelled,
        tx_hash_for=lambda block_idx, node: None,
    )
    result["summary"]["transactions"] = len(rows)
    return {"source": "upload", "mode": mode, "checkpoint": checkpoint,
            "threshold": threshold, "seed": None, **result}
