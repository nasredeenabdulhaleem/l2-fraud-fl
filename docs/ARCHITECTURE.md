# System Architecture Reference

This document describes the component boundaries and data flows of the prototype. It is the engineering companion to Chapter 3 (System Architecture Design) and underpins the phased build in `IMPLEMENTATION_GUIDE.md`.

## Three-Layer Architecture

The system is organised into three layers with a strict on-chain and off-chain boundary.

### Layer 1: Local node (off chain, per client)

Each participating Layer 2 node runs a local client agent. The agent ingests its own live transaction stream, constructs block-level transaction graphs, and trains a hybrid GraphSAGE-LSTM classifier. Raw transaction data never leaves this boundary. The only artefact that leaves the node is a compressed gradient update, and even that is committed on chain only as a fixed-size content hash.

Components: the data pipeline (`packages/data`), the model (`packages/models/gnn_lstm.py`), and the Flower client (`packages/fl/client.py`).

### Layer 2: Federation and aggregation (off chain coordinator plus on chain audit)

The Flower server (`packages/fl/server.py`) coordinates rounds, distributes the global model, and aggregates client updates using the selected strategy (FedProx or SCAFFOLD in `packages/fl/strategy.py`). Byzantine-robust filtering of client updates happens here, off chain, because it needs the actual gradient values.

The on-chain FLAggregator contract (`contracts/src/FLAggregator.sol`) governs the round lifecycle and records fixed-size commitments for tamper-evident audit. The coordinator opens a round on chain, clients (or the coordinator on their behalf) record their update commitments, and the coordinator finalises the round with a commitment to the newly aggregated global model.

### Layer 3: Application and demonstration (off chain)

The FastAPI telemetry service (`backend/main.py`) bridges federation callbacks and on-chain events into a single WebSocket event stream. The React dashboard (`frontend/`) consumes that stream to show connected nodes, round progress, active strategy, live fraud flags, and on-chain finalisation.

## Federated Round Sequence

1. Coordinator opens round r on chain with a commitment to the current global model.
2. Coordinator distributes the global model parameters to selected clients via Flower.
3. Each client trains locally on its shard. FedProx adds a proximal term against the global model; SCAFFOLD applies control-variate correction.
4. Each client returns its parameter update to the Flower server, and a fixed-size commitment to that update is recorded on chain.
5. The server aggregates the updates (with Byzantine filtering) into a new global model.
6. Coordinator finalises round r on chain with a commitment to the new global model.
7. The telemetry service streams every transition to the dashboard.

## Why the On-Chain and Off-Chain Split

On an optimistic rollup, execution and calldata are inexpensive relative to Layer 1, but persistent storage writes are still the dominant cost, and a model gradient is orders of magnitude too large to store as contract state. The architecture therefore confines the contract to coordination and audit: it stores only 32-byte commitments and small counters, all bounded fixed-size operations, so per-transaction gas is predictable and can be asserted in the Foundry test suite. The heavy numerical work stays off chain.

## Data Flow Summary

```
  live tx stream
       |
       v
  graph_builder  ->  per-block PyG snapshots (node feats, edge index, labels)
       |
       v
  GNN-LSTM  ->  local training  ->  parameter update
       |                                  |
       |                                  v
       |                         Flower server (FedProx / SCAFFOLD)
       |                                  |
       |                    aggregate + Byzantine filter
       |                                  |
       v                                  v
  32-byte commitment  ------------>  FLAggregator (on chain: open/submit/finalise)
                                          |
                                          v
                                 events -> FastAPI -> WebSocket -> React dashboard
```

## Interface Contracts

- Client to server: Flower `NumPyClient` (`get_parameters`, `fit`, `evaluate`). Parameters serialise as a list of numpy arrays in `state_dict` order.
- Coordinator to chain: `openRound(bytes32, uint64)`, `submitUpdate(bytes32)`, `finaliseRound(bytes32)`, plus views. See `packages/chain/aggregator_client.py`.
- Backend to frontend: JSON messages over WebSocket with a `type` field in {snapshot, round_opened, update_submitted, fraud_flagged, round_finalised}.
