# Design and Implementation of Blockchain Fraud Detection on Layer 2 Rollups using Federated Learning

## Technical Implementation Guide

Author: Abdulhaleem Nasredeen Hamza (FCP/CSC/22/1072)
Department of Computer Science, Federal University Dutse
Companion document to Chapter 4 (System Implementation and Testnet Deployment)

---

## 1. Purpose and Scope of this Document

This document is the engineering specification and build manual for the prototype that realises Objective 3 of the dissertation. It translates the system architecture defined in Chapter 3 into a concrete, reproducible implementation. It is organised as six sequential build phases, each mapped to a two-week Scrum-XP sprint, with a distinct and independently verifiable deliverable per phase.

The system is a proactive, privacy-preserving secondary verification layer that operates inside the Layer 2 rollup execution path. It detects semantic financial fraud (wash trading and flash loan exploitation) before a transaction batch achieves state finality on Ethereum Layer 1. It does this without any raw transaction data leaving the originating node, using federated learning to share only compressed gradient updates.

The three-step operational pipeline is:

1. Local training. Each L2 node trains a lightweight hybrid GraphSAGE-LSTM model on its own live transaction stream. Raw data never crosses the node boundary.
2. Parameter synchronisation. Each node compresses its local gradient update, publishes the update payload to off-chain storage, and commits a fixed-size cryptographic reference to the L2-native Aggregator smart contract.
3. Proactive inference. The aggregated global model scores incoming transaction payloads and flags malicious candidates before they are included in a committed state batch.

---

## 2. Technology Stack and Version Baseline

| Layer | Technology | Version target | Role |
|-------|-----------|----------------|------|
| Client language | Python | 3.10+ | Data pipeline, model, FL client agents |
| Graph and tensor stack | PyTorch, PyTorch Geometric | torch 2.2+, PyG 2.5+ | GraphSAGE spatial encoder, tensor ops |
| Sequence modelling | PyTorch nn.LSTM | bundled | Temporal encoder across blocks |
| Federated orchestration | Flower (flwr) | 1.8+ | Client-server federation, round control, strategy interface |
| Feature and graph tooling | Pandas, NumPy, Scikit-learn, NetworkX | current | Feature extraction, graph construction, metrics |
| Smart contract language | Solidity | 0.8.24 | L2-native Aggregator contract |
| Contract toolchain | Foundry (forge, cast, anvil) | current | Compile, gas profiling, trace testing, deployment |
| Supplementary testing | Hardhat | current | Integration testing across the JS bridge |
| Chain bridge | web3.py | 6.x | Python to contract interaction |
| Backend service | FastAPI, Uvicorn, websockets | current | Telemetry API, live event stream |
| Frontend | React, Vite, Recharts, ethers.js | React 18, Vite 5 | Live monitoring and demonstration dashboard |
| Target networks | Arbitrum Sepolia, Base Sepolia | testnet | Deployment and validation environment |

A hard formatting and engineering rule applies throughout: contracts store only fixed-size commitments, never raw model weights, because Layer 2 calldata and storage remain the dominant gas cost even at reduced L2 prices.

---

## 3. Repository Layout

```
l2-fraud-fl/
  docs/
    IMPLEMENTATION_GUIDE.md      this document
    ARCHITECTURE.md              component and data-flow reference
  contracts/                     Solidity and Foundry workspace
    foundry.toml
    src/FLAggregator.sol         L2-native coordination and audit contract
    test/FLAggregator.t.sol      gas-bounded and permission-edge unit tests
    script/Deploy.s.sol          testnet deployment script
  packages/
    data/                        data acquisition and graph construction
      elliptic_loader.py
      l2_simulator.py
      graph_builder.py
    models/                      hybrid model definition
      gnn_lstm.py
    fl/                          Flower client, server, strategies
      client.py
      server.py
      strategy.py                FedProx and SCAFFOLD strategies
      task.py                    shared train/eval/parameter helpers
    chain/                       Python bridge to the Aggregator contract
      aggregator_client.py
  backend/
    main.py                      FastAPI telemetry service with WebSocket feed
  frontend/                      React monitoring dashboard
    src/App.jsx
    ...
  requirements.txt
  pyproject.toml
  .env.example
```

The monorepo intentionally separates the on-chain workspace (Solidity plus Foundry), the Python federated learning core, the data pipeline, and the application layer. Each can be developed, tested, and versioned independently while sharing a single environment contract through `.env`.

---

## 4. Phased Build Plan Overview

| Phase | Sprint | Focus | Primary deliverable |
|-------|--------|-------|--------------------|
| 0 | Sprint 1 | Environment and repository scaffolding | Reproducible dev environment, compiling contract skeleton, passing smoke tests |
| 1 | Sprint 2 | Data acquisition and transaction-graph construction | Elliptic loader plus L2 fraud simulator producing PyG temporal graph snapshots |
| 2 | Sprint 3 | Local model development | Trained GraphSAGE-LSTM classifier with a validated single-node baseline |
| 3 | Sprint 4 | Federated integration via Flower | Multi-client FedProx and SCAFFOLD federation converging on non-IID partitions |
| 4 | Sprint 5 | L2-native smart contract aggregation | Deployed Aggregator on Arbitrum Sepolia and Base Sepolia with gas-bounded tests |
| 5 | Sprint 6 | Backend telemetry and frontend visualisation | FastAPI stream plus React dashboard demonstrating a live federated round end to end |

Each phase below states its objective, the engineering tasks, the exact artefacts produced, and the acceptance criteria used to close the sprint.

---

## 5. Phase 0: Environment and Repository Scaffolding

### 5.1 Objective

Establish a fully reproducible development environment and a compiling, test-covered skeleton across every layer, so that later phases add capability rather than fight configuration.

### 5.2 Engineering tasks

1. Create the Python virtual environment on Python 3.10 or later and install the pinned dependency set from `requirements.txt`. PyTorch Geometric wheels are index-sensitive, so installation order matters: install torch first, then the PyG companion wheels matched to the torch and CUDA build.
2. Install Foundry through `foundryup`. Confirm `forge`, `cast`, and `anvil` are on the path.
3. Initialise the Foundry workspace in `contracts/`, add the contract skeleton and a trivial passing test, and confirm `forge build` and `forge test` succeed.
4. Populate `.env.example` with the full set of environment keys: RPC endpoints for Arbitrum Sepolia and Base Sepolia, a deployer key placeholder, the deployed contract address placeholder, and the FL server address and port.
5. Add a `pyproject.toml` that registers `packages/` as an importable namespace so the data, models, fl, and chain modules resolve cleanly.

### 5.3 Artefacts produced

`requirements.txt`, `pyproject.toml`, `.env.example`, `contracts/foundry.toml`, a compiling `FLAggregator.sol` skeleton, and a passing smoke test.

### 5.4 Acceptance criteria

`forge test` passes. A one-line Python import of every package module succeeds without error. RPC connectivity to both testnets is confirmed with a `cast block-number` call against each endpoint.

---

## 6. Phase 1: Data Acquisition and Transaction-Graph Construction

### 6.1 Objective

Produce the training substrate: a stream of block-level transaction graphs with node features and fraud labels, drawn from the Elliptic dataset for realistic topology and augmented with a controllable L2 fraud simulator for the specific patterns of interest.

### 6.2 Rationale for the two-source strategy

Ground-truth-labelled fraud data native to Layer 2 transaction streams is not publicly available at usable scale. The dissertation resolves this with two complementary sources. The Elliptic Bitcoin dataset (Weber et al., 2019) supplies a large, realistically structured transaction graph with licit, illicit, and unknown labels, giving the spatial encoder authentic topological signal. A parametric L2 simulator built on NetworkX injects the two target fraud archetypes, wash-trading cycles and flash-loan star-bursts, with configurable base rates so that non-IID partitioning across clients can be controlled precisely for the federated experiments.

### 6.3 Engineering tasks

1. Implement `elliptic_loader.py` to download or ingest the Elliptic node feature matrix, edge list, and class map, resolve the tri-class labels into a binary illicit indicator, and expose the data as time-step-partitioned frames. The Elliptic dataset is already segmented into 49 discrete time steps, which map naturally onto the block-sequence abstraction the LSTM consumes.
2. Implement `l2_simulator.py` to synthesise directed transaction subgraphs containing wash-trading cycles (a small set of addresses transacting in a closed loop to fabricate volume) and flash-loan bursts (a single address opening and closing a large position across many counterparties within one block). Expose base-rate and value-distribution parameters so heterogeneous client partitions can be generated.
3. Implement `graph_builder.py` to convert either source into a sequence of `torch_geometric.data.Data` snapshots, one per block or time step, carrying node feature tensors, edge indices, and node-level labels. Provide a non-IID partitioner that splits the node population across a configurable number of clients with skewed fraud base rates.

### 6.4 Artefacts produced

`packages/data/elliptic_loader.py`, `packages/data/l2_simulator.py`, `packages/data/graph_builder.py`, and a serialised set of per-client temporal graph snapshots ready for training.

### 6.5 Acceptance criteria

The builder yields a list of PyG `Data` objects whose node feature dimension, edge index shape, and label vector are internally consistent. The non-IID partitioner produces client shards with measurably different fraud base rates (verified by printing per-client positive-class ratios). The simulator's injected fraud subgraphs are recoverable by label.

---

## 7. Phase 2: Local Model Development

### 7.1 Objective

Build and validate the per-node classifier: a hybrid model that fuses a spatial GraphSAGE encoder with a temporal LSTM encoder, trained and evaluated on a single client's data to establish a centralised baseline before federation.

### 7.2 Model rationale

Fraud on a rollup has two signatures that a single model family cannot capture alone. The spatial signature lives in the transaction graph topology within a block: wash trading appears as tight cycles, flash-loan exploitation as high-degree stars. GraphSAGE is chosen over a plain GCN because it aggregates over sampled neighbourhoods inductively, which means it generalises to nodes and addresses never seen during training. This is essential for a live L2 stream where new addresses appear continuously and a transductive model would be unusable at inference. The temporal signature lives in how block-level aggregate state evolves across the sequence of blocks. An LSTM captures this sequential dependency. The composition applies GraphSAGE per block to produce a spatial embedding, then feeds the sequence of block embeddings through the LSTM to produce a temporally informed representation for classification.

This is an architectural composition, not a modification of either base component's internal equations. GraphSAGE and the LSTM retain their standard formulations; the contribution is the pipeline that joins them for the L2 fraud task.

### 7.3 Engineering tasks

1. Implement `gnn_lstm.py` with a `GraphSAGE` spatial block (two `SAGEConv` layers with non-linearity and dropout) producing per-node spatial embeddings, followed by mean or attention pooling to a per-block embedding, followed by an `LSTM` over the block sequence, followed by a linear classification head.
2. Implement the training and evaluation helpers in `packages/fl/task.py` so that the identical train and evaluate functions are reused by both the single-node baseline and the federated clients. This guarantees that the federated result is measured on the same code path as the baseline.
3. Handle class imbalance with a weighted cross-entropy loss, since fraud is rare by construction.

### 7.4 Artefacts produced

`packages/models/gnn_lstm.py`, the shared `packages/fl/task.py`, and a saved single-node baseline checkpoint with recorded metrics.

### 7.5 Acceptance criteria

The single-node model trains without shape errors, loss decreases across epochs, and the model produces a defensible baseline on a held-out split (precision, recall, and F1 recorded for later comparison against the federated result).

---

## 8. Phase 3: Federated Integration via Flower

### 8.1 Objective

Coordinate multiple client models into a single global model using Flower, with two aggregation strategies implemented and compared under non-IID conditions: FedProx and SCAFFOLD.

### 8.2 Strategy rationale and mathematics

Naive federated averaging degrades badly when client data distributions differ, which is exactly the situation across heterogeneous L2 nodes serving different protocols and fraud base rates. Two remedies are implemented.

FedProx (Li et al., 2020) adds a proximal term to each client's local objective that penalises drift away from the current global model. The client minimises

```
    min_w   F_k(w) + (mu / 2) * || w - w_t ||^2
```

where `F_k(w)` is the local empirical loss, `w_t` is the global model at the start of round `t`, and `mu` is the proximal coefficient. The proximal term keeps heterogeneous local updates from diverging and stabilises aggregation under non-IID data.

SCAFFOLD (Karimireddy et al., 2020) corrects client drift directly using control variates. The server maintains a global control variate `c`, and each client maintains a local control variate `c_k`. The client local update becomes

```
    y  <-  y  -  eta_l * ( g_k(y)  -  c_k  +  c )
```

where `g_k(y)` is the local gradient and `eta_l` is the local learning rate. After local training the client refreshes its control variate and returns both the model delta and the control-variate delta. The correction term `(- c_k + c)` cancels the systematic component of client drift, giving faster and more stable convergence than plain averaging on non-IID partitions, at the cost of transmitting the extra control-variate state.

The comparison of these two strategies under identical L2-realistic non-IID partitions is a direct contribution of the work.

### 8.3 Engineering tasks

1. Implement `client.py` as a `flwr.client.NumPyClient` that wraps the Phase 2 model and the shared train and evaluate functions, exposing `get_parameters`, `fit`, and `evaluate`. The FedProx proximal term is applied inside `fit` by regularising against the received global parameters.
2. Implement `strategy.py` with a FedProx-configured strategy (built on Flower's `FedProx`) and a custom SCAFFOLD strategy that aggregates both model deltas and control-variate deltas.
3. Implement `server.py` to launch the Flower server with a configurable number of rounds, minimum client count, and selected strategy, and to expose per-round aggregated metrics for the telemetry layer.
4. Wire the client data source to the non-IID partitioner from Phase 1 so each simulated client trains on a distinct shard.

### 8.4 Artefacts produced

`packages/fl/client.py`, `packages/fl/server.py`, `packages/fl/strategy.py`, and a federated training run log showing per-round global metrics for both strategies.

### 8.5 Acceptance criteria

A federation of at least three clients on non-IID shards converges, the global model matches or approaches the centralised baseline from Phase 2, and both FedProx and SCAFFOLD complete a full multi-round run with metrics recorded for side-by-side comparison.

---

## 9. Phase 4: L2-Native Smart Contract Aggregation

### 9.1 Objective

Deploy the on-chain coordination and audit layer. The Aggregator contract governs federated rounds, accepts fixed-size commitments to client updates, and records the finalised global model reference, all within the gas envelope of an optimistic rollup.

### 9.2 On-chain design under L2 gas constraints

Storing raw gradient tensors on chain is infeasible. Even on Layer 2, where execution and calldata are far cheaper than mainnet, persistent storage writes remain the dominant cost, and a model update is far too large to store as state. The design therefore keeps all heavy data off chain and stores only what is needed for coordination and tamper-evident audit.

The contract records, per round, a fixed 32-byte commitment to each client's update payload (a content hash, with the payload itself held in off-chain storage referenced by a content identifier), the count of submissions, the round deadline, and, on finalisation, the 32-byte reference to the aggregated global model. Every write is a bounded, fixed-size operation. The heavy work of gradient aggregation and Byzantine-robust filtering happens off chain in the Flower server; the contract records which submissions were admitted so the aggregation is auditable. This division keeps per-transaction gas predictable and testable.

### 9.3 Engineering tasks

1. Implement `FLAggregator.sol`: coordinator role and client registry, round lifecycle (open, submit, finalise), fixed-size commitment storage, and events for every state transition so off-chain services can subscribe.
2. Enforce access control: only the coordinator opens and finalises rounds; only registered clients submit; submissions are rejected after the deadline and duplicate submissions per client per round are blocked.
3. Write the Foundry test suite `FLAggregator.t.sol` asserting correct behaviour, permission edge cases (unregistered client, non-coordinator finalisation, double submission, late submission), and explicit gas bounds on the hot paths (`submitUpdate`, `finaliseRound`) using `vm.snapshotGas` or gas measurement so a regression that inflates gas fails the build.
4. Implement `Deploy.s.sol` and deploy to Arbitrum Sepolia and Base Sepolia, recording the deployed addresses into `.env`.
5. Implement `packages/chain/aggregator_client.py` so the Flower server can open a round, read submitted commitments, and finalise on chain through web3.py.

### 9.4 Artefacts produced

`contracts/src/FLAggregator.sol`, `contracts/test/FLAggregator.t.sol`, `contracts/script/Deploy.s.sol`, `packages/chain/aggregator_client.py`, and two live testnet deployments.

### 9.5 Acceptance criteria

All Foundry tests pass including the gas-bound assertions. The contract is live on both testnets. The Python bridge can open a round, submit a commitment, and finalise, with the resulting transactions visible on the respective block explorers.

---

## 10. Phase 5: Backend Telemetry and Frontend Visualisation

### 10.1 Objective

Make the system demonstrable. A FastAPI service exposes federated round state and a live event stream over WebSockets; a React dashboard visualises connected nodes, round progress, aggregation strategy in use, live fraud flags, and on-chain finalisation, so the running system can be shown and defended.

### 10.2 Engineering tasks

1. Implement `backend/main.py`: REST endpoints for current federation state and round history, and a WebSocket endpoint that pushes events (client joined, round opened, update submitted, round finalised, fraud flagged) as they occur. Provide a demonstration mode that emits synthetic events so the UI renders end to end without a full federation running.
2. Bridge the Flower server and the on-chain events into the backend event bus so real round activity streams to the dashboard.
3. Implement the React dashboard: a federation overview (client count, active strategy, current round, global metrics), a live round timeline, a per-client submission grid, a streaming fraud-alert feed, and an on-chain panel showing the latest finalised model reference and its transaction hash with a link to the explorer.
4. Use Recharts for the metric trend charts and ethers.js for reading the Aggregator contract state directly from the browser as an independent verification of the backend feed.

### 10.3 Artefacts produced

`backend/main.py`, the React application under `frontend/`, and a runbook for starting the full stack (contract listener, Flower server, backend, frontend).

### 10.4 Acceptance criteria

Starting the stack shows a live dashboard that reflects a real or demonstration federated round from client join through on-chain finalisation, with fraud flags appearing in the alert feed and metric trends updating per round.

---

## 11. End-to-End Runbook

1. Deploy the contract: from `contracts/`, run the Foundry deploy script against the chosen testnet and copy the address into `.env`.
2. Prepare data: from the repo root, run the data pipeline to generate per-client temporal graph snapshots.
3. Start the backend: launch the FastAPI telemetry service.
4. Start the Flower server with the chosen strategy (FedProx or SCAFFOLD) and the target round count.
5. Start the simulated clients, each bound to a non-IID shard.
6. Start the frontend and open the dashboard to watch the federation and on-chain finalisation live.

---

## 12. Mapping to Dissertation Objectives

Objective 2 (System Architecture Design) is realised as the concrete component boundaries in Phases 2, 3, and 4 and documented in `ARCHITECTURE.md`. Objective 3 (System Implementation and Testnet Deployment) is realised in full across Phases 0 through 5, with the two testnet deployments and the live dashboard as its terminal deliverables. Objectives 4 (quantitative evaluation) and 5 (adversarial robustness) remain scoped as future work; the telemetry layer built in Phase 5 is deliberately structured to feed those later measurement campaigns without rework.

---

## 13. Verified References

Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., and Suresh, A. T. (2020). SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. Proceedings of the 37th International Conference on Machine Learning, PMLR 119, 5132 to 5143.

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. (2020). Federated Optimization in Heterogeneous Networks. Proceedings of Machine Learning and Systems, 2, 429 to 450.

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., and Aguera y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, PMLR 54, 1273 to 1282.

Hamilton, W. L., Ying, R., and Leskovec, J. (2017). Inductive Representation Learning on Large Graphs (GraphSAGE). Advances in Neural Information Processing Systems 30.

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., and Leiserson, C. E. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. KDD Workshop on Anomaly Detection in Finance.
