# Blockchain Fraud Detection on Layer 2 Rollups using Federated Learning

A proactive, privacy-preserving secondary verification layer that detects semantic financial fraud (wash trading and flash-loan exploitation) inside the Layer 2 rollup execution path, before transaction batches achieve state finality on Ethereum. Nodes collaboratively train a hybrid GraphSAGE-LSTM model via federated learning, sharing only compressed gradient updates; raw transaction data never leaves a node.

Undergraduate dissertation prototype. Author: Abdulhaleem Nasredeen Hamza (FCP/CSC/22/1072), Department of Computer Science, Federal University Dutse.

## What is in this repository

| Path | Contents |
|------|----------|
| `docs/IMPLEMENTATION_GUIDE.md` | The full six-phase build manual (start here) |
| `docs/ARCHITECTURE.md` | Component boundaries and data flows |
| `contracts/` | Solidity FLAggregator contract, Foundry tests, deploy script |
| `packages/data/` | Elliptic loader, L2 fraud simulator, PyG graph builder, non-IID partitioner |
| `packages/models/` | Hybrid GraphSAGE-LSTM classifier |
| `packages/fl/` | Flower client, server, FedProx and SCAFFOLD strategies, shared task helpers |
| `packages/chain/` | web3.py bridge to the on-chain Aggregator |
| `backend/` | FastAPI telemetry service with a live WebSocket stream and demo mode |
| `frontend/` | React monitoring dashboard (Recharts, ethers.js) |

## Technology stack

Python 3.10+, PyTorch, PyTorch Geometric, Flower (FLWR), Solidity 0.8.24 with Foundry, web3.py, FastAPI with WebSockets, React with Vite. Targets: Arbitrum Sepolia and Base Sepolia testnets.

## Quickstart

### 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env    # then fill in RPC URLs and a testnet key
```

Note: install PyTorch first if your platform needs a specific wheel, then the PyTorch Geometric companion wheels matched to your torch and CUDA build.

### 2. Contracts (Foundry)

```bash
cd contracts
forge install foundry-rs/forge-std
forge build
forge test -vv           # runs functional, permission, and gas-bound tests
```

Deploy to a testnet:

```bash
forge script script/Deploy.s.sol:Deploy \
  --rpc-url $ARBITRUM_SEPOLIA_RPC --broadcast --private-key $DEPLOYER_PRIVATE_KEY
```

Copy the printed address into `FLAGGREGATOR_ADDRESS_ARBITRUM` in `.env`.

### 3. Run the full stack (demonstration)

Terminal A, telemetry backend (demo mode emits synthetic rounds so the UI renders without a full federation):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Terminal B, frontend dashboard:

```bash
cd frontend && npm install && npm run dev
# open http://localhost:5173
```

### 4. Run a real federation

Terminal A, Flower server (pick a strategy):

```bash
python -m packages.fl.server --strategy fedprox --rounds 10 --min-clients 3
# or: --strategy scaffold
```

Terminals B, C, D, one client each on its own non-IID shard:

```bash
python -m packages.fl.client --client-id 0 --num-clients 3 --server 127.0.0.1:8080
python -m packages.fl.client --client-id 1 --num-clients 3 --server 127.0.0.1:8080
python -m packages.fl.client --client-id 2 --num-clients 3 --server 127.0.0.1:8080
```

Set `BACKEND_DEMO_MODE=false` in `.env` when wiring the real Flower callbacks and the on-chain event listener into the telemetry backend.

## Build phases

The work is structured as six sprints, each with a distinct deliverable: environment scaffolding, data and graph construction, local model development, federated integration via Flower, L2-native smart contract aggregation, and backend plus frontend visualisation. See `docs/IMPLEMENTATION_GUIDE.md` for the detailed phase-by-phase specification and acceptance criteria.

## Scope

This prototype delivers the design and full implementation (dissertation Objectives 2 and 3), deployed on testnet with a live dashboard. Formal quantitative evaluation and adversarial robustness benchmarking are scoped as future work (Objectives 4 and 5); the telemetry layer is built to feed those later measurement campaigns without rework.
