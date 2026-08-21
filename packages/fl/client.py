"""Flower client agent.

Wraps the hybrid GNN-LSTM model and the shared task helpers. Each client trains on
its own non-IID shard and returns only model parameters, never raw data, satisfying
the privacy constraint that transaction data never crosses the node boundary.

Two aggregation regimes are supported from the client side:
  FedProx: the proximal term is applied inside fit by passing the received global
           parameters and the proximal coefficient mu into task.train.
  SCAFFOLD: the client additionally maintains a local control variate and returns
            its delta alongside the model delta (see strategy.py for aggregation).
"""

from __future__ import annotations

import argparse

import flwr as fl
import numpy as np
import torch

from packages.data.graph_builder import stratified_split
from packages.fl import task


class FraudFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_snapshots: list,
        val_snapshots: list,
        proximal_mu: float = 0.1,
        local_epochs: int = 3,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model
        self.train_snapshots = train_snapshots
        self.val_snapshots = val_snapshots
        self.proximal_mu = proximal_mu
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        # SCAFFOLD's local control variate. Persists across rounds because this
        # client object lives for the whole run (one long-running process per
        # client under fl.client.start_client), lazily zero-initialised once the
        # model shapes are known on the first SCAFFOLD round.
        self.c_local: list[np.ndarray] | None = None

    def get_parameters(self, config):
        return task.get_parameters(self.model)

    def fit(self, parameters, config):
        lr = float(config.get("lr", self.lr))
        epochs = int(config.get("local_epochs", self.local_epochs))

        if config.get("scaffold"):
            # strategy.py's ScaffoldStrategy.configure_fit concatenates the
            # global control variate c onto parameters (FitIns.config can't
            # carry arrays), so split it back apart here.
            n = len(parameters) // 2
            model_params, c_global = parameters[:n], parameters[n:]
            mu = 0.0  # FedProx's proximal term is a separate strategy, not combined here.
        else:
            model_params, c_global = parameters, None
            mu = float(config.get("proximal_mu", self.proximal_mu))

        task.set_parameters(self.model, model_params)

        if c_global is not None and self.c_local is None:
            self.c_local = [np.zeros_like(p) for p in model_params]

        metrics = task.train(
            self.model,
            self.train_snapshots,
            epochs=epochs,
            lr=lr,
            proximal_mu=mu,
            global_params=model_params if mu > 0 else None,
            control_variate=(self.c_local, c_global) if c_global is not None else None,
            device=self.device,
        )
        new_params = task.get_parameters(self.model)

        if c_global is not None:
            # SCAFFOLD option-II control-variate update:
            #   c_k_new = c_k - c + (w_global - w_local) / (K * eta_l)
            k = metrics["num_local_steps"]
            c_local_new = [
                c_k - c_g + (w_g - w_l) / (k * lr)
                for c_k, c_g, w_g, w_l in zip(self.c_local, c_global, model_params, new_params)
            ]
            c_delta = [new - old for new, old in zip(c_local_new, self.c_local)]
            self.c_local = c_local_new
            return new_params + c_delta, metrics["num_examples"], {"loss": metrics["loss"]}

        return new_params, metrics["num_examples"], {"loss": metrics["loss"]}

    def evaluate(self, parameters, config):
        task.set_parameters(self.model, parameters)
        metrics = task.evaluate(self.model, self.val_snapshots, device=self.device)
        return metrics["loss"], metrics["num_examples"], {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }


def main():
    """Launch a single client bound to a non-IID shard produced by the data pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--proximal-mu", type=float, default=0.1)
    args = parser.parse_args()

    # Build the client's data shard. Uses the L2 simulator by default; swap in the
    # Elliptic pipeline by importing elliptic_to_snapshots instead.
    from packages.data.l2_simulator import L2FraudSimulator, SimConfig
    from packages.data.graph_builder import simulator_to_snapshots, partition_non_iid

    blocks = L2FraudSimulator(SimConfig(seed=7)).generate()
    snapshots = simulator_to_snapshots(blocks)
    shards = partition_non_iid(snapshots, num_clients=args.num_clients)
    my_stream = shards[args.client_id]
    train_s, val_s = stratified_split(my_stream)

    in_dim = my_stream[0].x.size(1)
    model = task.build_model(in_dim=in_dim)

    client = FraudFLClient(
        model=model,
        train_snapshots=train_s,
        val_snapshots=val_s,
        proximal_mu=args.proximal_mu,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
