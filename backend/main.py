"""FastAPI telemetry service.

Exposes REST endpoints for current federation state and round history, and a
WebSocket endpoint that pushes events as they occur (client joined, round opened,
update submitted, round finalised, fraud flagged). A demonstration mode emits
synthetic events so the dashboard renders end to end without a full federation
running, which is useful for the viva and for UI development.

Run:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from packages.models.infer import ScoringError, available_checkpoints, score_transaction

_KNOWN_EVENT_TYPES = {
    "round_opened", "update_submitted", "fraud_flagged", "round_finalised",
}

DEMO_MODE = os.getenv("BACKEND_DEMO_MODE", "true").lower() == "true"
DEMO_NETWORK = os.getenv("TARGET_NETWORK", "arbitrum")


# ----------------------------------------------------------------------------
# In-memory federation state (a real deployment would source this from the
# Flower server callbacks and the on-chain event listener)
# ----------------------------------------------------------------------------

class FederationState:
    def __init__(self):
        self.strategy = os.getenv("FL_STRATEGY", "fedprox")
        self.current_round = 0
        self.total_rounds = int(os.getenv("FL_NUM_ROUNDS", 10))
        self.clients = [
            {"id": i, "address": f"0x{random.randrange(16**8):08x}", "fraud_rate": 0.0,
             "submitted": False}
            for i in range(int(os.getenv("FL_MIN_CLIENTS", 3)))
        ]
        self.round_history = []          # list of {round, precision, recall, f1, tx_hash}
        self.fraud_alerts = []           # recent flagged transactions

    def snapshot(self) -> dict:
        return {
            "strategy": self.strategy,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "clients": self.clients,
            "round_history": self.round_history[-50:],
            "fraud_alerts": self.fraud_alerts[-30:],
        }

    def apply_event(self, event_type: str, data: dict) -> None:
        """Update persisted state the same way for demo and real events, so a
        newly connecting client's initial snapshot reflects whichever source
        (demo_loop or a real FL run via /api/events) is actually driving it.
        """
        if event_type == "round_opened":
            self.current_round = data.get("round", self.current_round)
            for c in self.clients:
                c["submitted"] = False

        elif event_type == "update_submitted":
            client_id = data.get("client_id")
            client = next((c for c in self.clients if c["id"] == client_id), None)
            if client is None:
                client = {"id": client_id, "address": data.get("address", ""),
                          "fraud_rate": 0.0, "submitted": False}
                self.clients.append(client)
            client["address"] = data.get("address", client["address"])
            client["fraud_rate"] = data.get("fraud_rate", client["fraud_rate"])
            client["submitted"] = True

        elif event_type == "fraud_flagged":
            self.fraud_alerts.append(data)

        elif event_type == "round_finalised":
            self.round_history.append(data)


state = FederationState()


# ----------------------------------------------------------------------------
# WebSocket connection manager
# ----------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        await ws.send_json({"type": "snapshot", "data": state.snapshot()})

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ----------------------------------------------------------------------------
# Public API for the FL server / chain listener to push real events
# ----------------------------------------------------------------------------

async def emit(event_type: str, payload: dict):
    """Called by demo_loop or a real event source (via POST /api/events) to update
    persisted state and push it to connected dashboards. Centralising the state
    update here means a newly connecting client's initial snapshot is correct
    regardless of which source is driving events.
    """
    state.apply_event(event_type, payload)
    await manager.broadcast({"type": event_type, "data": payload, "ts": time.time()})


# ----------------------------------------------------------------------------
# Demonstration event generator
# ----------------------------------------------------------------------------

_FRAUD_TYPES = ["wash", "flash"]


async def demo_loop():
    """Emit a plausible federated round every few seconds for demonstration."""
    await asyncio.sleep(1.0)
    while True:
        if state.current_round >= state.total_rounds:
            await asyncio.sleep(5.0)
            state.current_round = 0
            state.round_history.clear()

        rnd = state.current_round + 1

        # Round opens.
        base_ref = f"0x{random.randrange(16**16):016x}"
        await emit("round_opened", {"round": rnd, "base_ref": base_ref})
        await asyncio.sleep(1.2)

        # Clients submit updates.
        for c in state.clients:
            fraud_rate = round(random.uniform(0.01, 0.12), 3)
            await emit("update_submitted", {
                "round": rnd, "client_id": c["id"], "address": c["address"],
                "fraud_rate": fraud_rate,
            })
            await asyncio.sleep(0.4)

        # Occasionally flag a fraudulent transaction pre-commitment.
        if random.random() < 0.7:
            alert = {
                "round": rnd,
                "tx": f"0x{random.randrange(16**32):032x}",
                "fraud_type": random.choice(_FRAUD_TYPES),
                "score": round(random.uniform(0.82, 0.99), 3),
            }
            await emit("fraud_flagged", alert)

        # Round finalises with improving metrics as rounds progress.
        base = min(0.95, 0.55 + 0.03 * rnd + random.uniform(-0.02, 0.02))
        record = {
            "round": rnd,
            "precision": round(min(0.98, base + random.uniform(0, 0.03)), 3),
            "recall": round(min(0.97, base - random.uniform(0, 0.05)), 3),
            "f1": round(base, 3),
            "tx_hash": f"0x{random.randrange(16**32):032x}",
            "network": DEMO_NETWORK,
        }
        await emit("round_finalised", record)
        await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if DEMO_MODE:
        task = asyncio.create_task(demo_loop())
    yield
    if task:
        task.cancel()


app = FastAPI(title="L2 Fraud FL Telemetry", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class StateResponse(BaseModel):
    strategy: str
    current_round: int
    total_rounds: int
    clients: list
    round_history: list
    fraud_alerts: list


@app.get("/api/state", response_model=StateResponse)
def get_state():
    return state.snapshot()


@app.get("/api/health")
def health():
    return {"ok": True, "demo": DEMO_MODE}


class EventIn(BaseModel):
    type: str
    data: dict


@app.post("/api/events")
async def post_event(event: EventIn):
    """Ingestion point for real events from the FL server / chain bridge
    (packages/fl/telemetry.py's TelemetryClient posts here). Runs as a separate
    process from the FL server, so this HTTP hop is the bridge between them.
    """
    if event.type not in _KNOWN_EVENT_TYPES:
        return {"ok": False, "error": f"unknown event type: {event.type}"}
    await emit(event.type, event.data)
    return {"ok": True}


# ----------------------------------------------------------------------------
# Transaction scoring (ad-hoc playground, independent of the FL telemetry
# state above -- this scores a single hand-built transaction context against a
# trained checkpoint rather than replaying federation round events)
# ----------------------------------------------------------------------------

class ScoreEdge(BaseModel):
    src: str
    dst: str
    value: float


class ScoreRequest(BaseModel):
    checkpoint: str
    target: str
    edges: list[ScoreEdge]


@app.get("/api/checkpoints")
def list_checkpoints():
    return {"checkpoints": available_checkpoints()}


@app.post("/api/score")
def score(req: ScoreRequest):
    try:
        return score_transaction(
            checkpoint_name=req.checkpoint,
            edges_in=[e.model_dump() for e in req.edges],
            target=req.target,
        )
    except ScoringError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive; client is not required to send
    except WebSocketDisconnect:
        manager.disconnect(ws)
