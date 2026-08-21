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
from packages.fl.telemetry import TelemetryClient, TelemetryStrategy


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
    parser.add_argument("--on-chain", action="store_true",
                         help="open/finalise each round on FLAggregator as the coordinator")
    parser.add_argument("--chain-network", choices=["arbitrum", "base"], default="arbitrum")
    parser.add_argument("--no-telemetry", action="store_true",
                         help="skip posting round events to the backend dashboard")
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

    chain_client = None
    if args.on_chain:
        from packages.chain.aggregator_client import AggregatorClient

        chain_client = AggregatorClient(network=args.chain_network)
        print(f"on-chain: coordinator {chain_client.acct.address}")

    # Compute each client's real dev-account address and local fraud rate for
    # the dashboard, using the same seed/num_clients partitioning client.py
    # uses independently -- this only reads shard statistics, it doesn't need
    # the clients to be connected yet.
    from packages.chain.dev_accounts import client_address
    from packages.data.graph_builder import client_fraud_rates as compute_fraud_rates
    from packages.data.graph_builder import partition_non_iid, simulator_to_snapshots
    from packages.data.l2_simulator import L2FraudSimulator, SimConfig

    blocks = L2FraudSimulator(SimConfig(seed=7)).generate()
    shards = partition_non_iid(simulator_to_snapshots(blocks), num_clients=args.min_clients)
    fraud_rates = compute_fraud_rates(shards)
    client_addresses = {i: client_address(i) for i in range(args.min_clients)}
    client_rates = {i: fraud_rates[i] for i in range(args.min_clients)}

    strategy = TelemetryStrategy(
        inner=strategy,
        telemetry=None if args.no_telemetry else TelemetryClient(),
        chain_client=chain_client,
        client_addresses=client_addresses,
        client_fraud_rates=client_rates,
        network=args.chain_network,
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
