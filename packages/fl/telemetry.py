"""Bridges a live FL run to the backend dashboard and the on-chain Aggregator.

The Flower server and the FastAPI backend (backend/main.py) are separate
processes, so TelemetryClient posts events over HTTP to backend/main.py's
POST /api/events, which forwards them to the WebSocket-connected dashboard
via its existing emit() -- the event shapes match exactly what demo_loop
already synthesizes, so the frontend needs no changes to consume real ones.

TelemetryStrategy wraps any Flower Strategy (FedProx built-in or
ScaffoldStrategy), delegating all actual aggregation logic unchanged, while
driving the on-chain round lifecycle (open_round at the start of fit,
finalise_round after evaluate aggregates) and emitting telemetry around it.
One implementation works for both strategies -- server.py just wraps
whichever one it builds.
"""

from __future__ import annotations

import logging
import os

import requests
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from packages.chain.aggregator_client import model_ref

logger = logging.getLogger(__name__)


class TelemetryClient:
    """Posts events to the backend. Never raises -- a dead/unreachable backend
    should not interrupt an otherwise-working FL run; telemetry is
    observability, not the critical path.
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.getenv("BACKEND_API_URL", "http://localhost:8000")

    def emit(self, event_type: str, data: dict) -> None:
        try:
            requests.post(
                f"{self.base_url}/api/events",
                json={"type": event_type, "data": data},
                timeout=5,
            )
        except requests.RequestException:
            logger.warning("telemetry post failed (backend unreachable?), continuing",
                            exc_info=True)


class TelemetryStrategy(Strategy):
    def __init__(
        self,
        inner: Strategy,
        telemetry: TelemetryClient | None = None,
        chain_client=None,
        client_addresses: dict[int, str] | None = None,
        client_fraud_rates: dict[int, float] | None = None,
        network: str = "arbitrum",
    ):
        self.inner = inner
        self.telemetry = telemetry
        self.chain_client = chain_client
        self.client_addresses = client_addresses or {}
        self.client_fraud_rates = client_fraud_rates or {}
        self.network = network
        self._last_agg_params: Parameters | None = None

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self.inner.initialize_parameters(client_manager)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        base_ref = model_ref(parameters_to_ndarrays(parameters))
        if self.chain_client is not None:
            try:
                self.chain_client.open_round(base_ref)
            except Exception:
                logger.warning("on-chain open_round failed, continuing without it",
                                exc_info=True)
        if self.telemetry is not None:
            self.telemetry.emit("round_opened", {
                "round": server_round, "base_ref": "0x" + base_ref.hex(), "network": self.network,
            })
        return self.inner.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results, failures):
        agg_params, agg_metrics = self.inner.aggregate_fit(server_round, results, failures)
        self._last_agg_params = agg_params

        if self.telemetry is not None:
            for _, fit_res in results:
                client_id = fit_res.metrics.get("client_id")
                self.telemetry.emit("update_submitted", {
                    "round": server_round,
                    "client_id": client_id,
                    "address": self.client_addresses.get(client_id, ""),
                    "fraud_rate": self.client_fraud_rates.get(client_id, 0.0),
                })
        return agg_params, agg_metrics

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        return self.inner.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round: int, results, failures):
        loss, metrics = self.inner.aggregate_evaluate(server_round, results, failures)

        tx_hash = ""
        if self.chain_client is not None and self._last_agg_params is not None:
            try:
                global_ref = model_ref(parameters_to_ndarrays(self._last_agg_params))
                receipt = self.chain_client.finalise_round(global_ref)
                tx_hash = receipt.transactionHash.hex()
            except Exception:
                logger.warning("on-chain finalise_round failed, continuing without it",
                                exc_info=True)

        if self.telemetry is not None:
            record: dict[str, Scalar] = {
                "round": server_round, "tx_hash": tx_hash, "network": self.network,
            }
            record.update(metrics)
            self.telemetry.emit("round_finalised", record)
        return loss, metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        return self.inner.evaluate(server_round, parameters)
