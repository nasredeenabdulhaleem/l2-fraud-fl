"""Aggregation strategies: FedProx and SCAFFOLD.

Naive federated averaging degrades under the non-IID data of heterogeneous L2 nodes.
Two remedies are implemented and compared under identical partitions.

FedProx (Li et al., 2020): each client minimises its local loss plus a proximal
term (mu / 2) || w - w_t ||^2 that penalises drift from the global model w_t. The
proximal term is applied client-side (see task.train); the server just distributes
mu and averages, so Flower's built-in FedProx strategy suffices.

SCAFFOLD (Karimireddy et al., 2020): corrects client drift with control variates.
The server holds a global control variate c; each client holds c_k. The local step is
    y  <-  y  -  eta_l ( g_k(y) - c_k + c ),
and clients return both the model delta and the control-variate delta. The server
aggregates both. This cancels the systematic drift component, giving faster and more
stable convergence on non-IID data at the cost of extra transmitted state.

The SCAFFOLD strategy below aggregates model and control-variate deltas. It is a
compact reference implementation intended for the dissertation's comparative study;
production hardening (secure aggregation, compression) is out of scope here.
"""

from __future__ import annotations

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx


def make_fedprox_strategy(
    initial_parameters: NDArrays,
    proximal_mu: float = 0.1,
    min_clients: int = 3,
) -> FedProx:
    """Flower's built-in FedProx. The proximal term is enforced client-side."""
    return FedProx(
        proximal_mu=proximal_mu,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        min_evaluate_clients=min_clients,
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        on_fit_config_fn=lambda rnd: {"proximal_mu": proximal_mu, "server_round": rnd},
    )


class ScaffoldStrategy(FedAvg):
    """SCAFFOLD server: aggregates model deltas and updates the global control variate.

    Convention: a client returns concatenated arrays [model_params..., c_delta...].
    The first half is the updated model; the second half is the client's control
    variate delta. The server averages model params (FedAvg core) and accumulates
    the averaged control-variate deltas into the global control variate c.
    """

    def __init__(self, initial_parameters: NDArrays, global_lr: float = 1.0, **kwargs):
        super().__init__(
            initial_parameters=ndarrays_to_parameters(initial_parameters), **kwargs
        )
        # Global control variate initialised to zeros with the model's shapes.
        self.c_global: list[np.ndarray] = [np.zeros_like(p) for p in initial_parameters]
        self.global_lr = global_lr
        self._num_params = len(initial_parameters)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Send the global control variate c alongside the model, like FedAvg does.

        FitIns.config only accepts scalars, not arrays, so c_global travels
        concatenated onto parameters (mirroring the convention aggregate_fit
        already expects on the return path: [model_params..., c_delta...]).
        The client splits it back apart using num_params from the "scaffold"
        config flag.
        """
        config = {"scaffold": True, "server_round": server_round}
        model_params = parameters_to_ndarrays(parameters)
        combined = ndarrays_to_parameters(model_params + self.c_global)
        fit_ins = FitIns(combined, config)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ):
        if not results:
            return None, {}

        model_updates: list[tuple[NDArrays, int]] = []
        c_deltas: list[NDArrays] = []

        for _, fit_res in results:
            flat = parameters_to_ndarrays(fit_res.parameters)
            model_params = flat[: self._num_params]
            c_delta = flat[self._num_params :]
            model_updates.append((model_params, fit_res.num_examples))
            if c_delta:
                c_deltas.append(c_delta)

        # Weighted average of model parameters (standard FedAvg core).
        total = sum(n for _, n in model_updates)
        agg_model = [
            sum(params[i] * n for params, n in model_updates) / total
            for i in range(self._num_params)
        ]

        # Update the global control variate by the mean control-variate delta.
        if c_deltas:
            mean_c_delta = [
                np.mean([cd[i] for cd in c_deltas], axis=0)
                for i in range(len(c_deltas[0]))
            ]
            self.c_global = [
                cg + self.global_lr * dc for cg, dc in zip(self.c_global, mean_c_delta)
            ]

        return ndarrays_to_parameters(agg_model), {"server_round": server_round}
