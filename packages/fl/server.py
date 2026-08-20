"""Flower server.

Launches the federation with a configurable strategy (FedProx or SCAFFOLD), round
count, and minimum client count. Aggregated per-round metrics are collected via a
metrics-aggregation callback so the telemetry layer (backend/main.py) can stream
them to the dashboard. The server is also the natural place to bridge to the
on-chain Aggregator: on each round it can open a round, and on finalisation commit
the aggregated global-model reference (see packages/chain/aggregator_client.py).
"""

from __future__ import annotations

import argparse
import os

import flwr as fl
from flwr.common import Metrics

from packages.fl import task
from packages.fl.strategy import make_fedprox_strategy, ScaffoldStrategy


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate client evaluate metrics weighted by example count."""
    total = sum(n for n, _ in metrics) or 1
    agg = {}
    for key in ("precision", "recall", "f1"):
        agg[key] = sum(n * m.get(key, 0.0) for n, m in metrics) / total
    return agg


def build_initial_model(in_dim: int):
    model = task.build_model(in_dim=in_dim)
    return task.get_parameters(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=int(os.getenv("FL_NUM_ROUNDS", 10)))
    parser.add_argument("--min-clients", type=int, default=int(os.getenv("FL_MIN_CLIENTS", 3)))
    parser.add_argument("--strategy", choices=["fedprox", "scaffold"],
                        default=os.getenv("FL_STRATEGY", "fedprox"))
    parser.add_argument("--proximal-mu", type=float, default=0.1)
    parser.add_argument("--in-dim", type=int, default=4,
                        help="node feature dim: 4 for the L2 simulator")
    parser.add_argument("--address", default=os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8080"))
    args = parser.parse_args()

    initial = build_initial_model(args.in_dim)

    if args.strategy == "fedprox":
        strategy = make_fedprox_strategy(
            initial_parameters=initial,
            proximal_mu=args.proximal_mu,
            min_clients=args.min_clients,
        )
        strategy.evaluate_metrics_aggregation_fn = weighted_average
    else:
        strategy = ScaffoldStrategy(
            initial_parameters=initial,
            min_fit_clients=args.min_clients,
            min_available_clients=args.min_clients,
            min_evaluate_clients=args.min_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    print(f"Starting Flower server: strategy={args.strategy} rounds={args.rounds} "
          f"min_clients={args.min_clients}")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
