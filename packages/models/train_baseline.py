"""Single-node centralised baseline.

Trains the same GraphSAGE-LSTM model on the full (unpartitioned) dataset using
the identical train/evaluate code path the federated clients use (packages/fl/task.py),
so the federated result can be measured against a real baseline rather than an
assumed one. Saves a checkpoint with recorded metrics for that later comparison.

Run:
  python -m packages.models.train_baseline --source simulator --epochs 20
  python -m packages.models.train_baseline --source elliptic --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from packages.data.graph_builder import stratified_split
from packages.fl import task

_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def _load_snapshots(source: str) -> list:
    if source == "simulator":
        from packages.data.l2_simulator import L2FraudSimulator, SimConfig
        from packages.data.graph_builder import simulator_to_snapshots

        blocks = L2FraudSimulator(SimConfig(seed=7)).generate()
        return simulator_to_snapshots(blocks)

    from packages.data.elliptic_loader import EllipticLoader
    from packages.data.graph_builder import elliptic_to_snapshots

    return elliptic_to_snapshots(EllipticLoader().load())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["simulator", "elliptic"], default="simulator")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default=str(_CHECKPOINT_DIR / "baseline.pt"))
    args = parser.parse_args()

    snapshots = _load_snapshots(args.source)
    train_s, val_s = stratified_split(snapshots)
    print(f"source={args.source} total_blocks={len(snapshots)} "
          f"train_blocks={len(train_s)} val_blocks={len(val_s)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = snapshots[0].x.size(1)
    model = task.build_model(in_dim=in_dim)

    train_metrics = task.train(model, train_s, epochs=args.epochs, lr=args.lr, device=device)
    val_metrics = task.evaluate(model, val_s, device=device)

    print(f"train_loss={train_metrics['loss']:.4f}")
    print(f"val_loss={val_metrics['loss']:.4f} "
          f"precision={val_metrics['precision']:.4f} "
          f"recall={val_metrics['recall']:.4f} "
          f"f1={val_metrics['f1']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "source": args.source,
        "metrics": {
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
        },
    }, out_path)
    print(f"checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
