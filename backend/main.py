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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DEMO_MODE = os.getenv("BACKEND_DEMO_MODE", "true").lower() == "true"


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
    """Called by the FL server callbacks or the chain listener to push a real event."""
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

        state.current_round += 1
        rnd = state.current_round

        # Round opens.
        base_ref = f"0x{random.randrange(16**16):016x}"
        for c in state.clients:
            c["submitted"] = False
        await emit("round_opened", {"round": rnd, "base_ref": base_ref})
        await asyncio.sleep(1.2)

        # Clients submit updates.
        for c in state.clients:
            c["fraud_rate"] = round(random.uniform(0.01, 0.12), 3)
            c["submitted"] = True
            await emit("update_submitted", {
                "round": rnd, "client_id": c["id"], "address": c["address"],
                "fraud_rate": c["fraud_rate"],
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
            state.fraud_alerts.append(alert)
            await emit("fraud_flagged", alert)

        # Round finalises with improving metrics as rounds progress.
        base = min(0.95, 0.55 + 0.03 * rnd + random.uniform(-0.02, 0.02))
        record = {
            "round": rnd,
            "precision": round(min(0.98, base + random.uniform(0, 0.03)), 3),
            "recall": round(min(0.97, base - random.uniform(0, 0.05)), 3),
            "f1": round(base, 3),
            "tx_hash": f"0x{random.randrange(16**32):032x}",
        }
        state.round_history.append(record)
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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive; client is not required to send
    except WebSocketDisconnect:
        manager.disconnect(ws)
