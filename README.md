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
| `packages/models/` | Hybrid GraphSAGE-LSTM classifier, single-node baseline trainer (`train_baseline.py`), ad-hoc scorer (`infer.py`), block-stream detector (`detector.py`) |
| `packages/fl/` | Flower client, server, FedProx and SCAFFOLD strategies, shared task helpers, telemetry bridge |
| `packages/chain/` | web3.py bridge to the on-chain Aggregator, deterministic client dev accounts, registration/smoke-test scripts |
| `backend/` | FastAPI telemetry service with a live WebSocket stream, demo mode, a real-event ingestion endpoint, and an ad-hoc transaction-scoring endpoint |
| `frontend/` | React dashboard: a Training tab (live FL monitoring), a Fraud Detection tab (sweep a simulated block stream and inspect every flag), and a Test a Transaction tab (score a hand-built transaction) |
| `notebooks/` | `train_colab.ipynb` — trains the baseline and federated models on a Colab GPU runtime |

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

Note: install PyTorch first if your platform needs a specific wheel, then the PyTorch Geometric companion wheels matched to your torch and CUDA build. No GPU handy? `notebooks/train_colab.ipynb` runs the same training (baseline + federated) on a free Colab GPU runtime and hands back checkpoints to drop into `packages/models/checkpoints/`.

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

Set `BACKEND_DEMO_MODE=false` in `.env` so the backend shows real round data instead of synthetic demo events (the backend from step 3 must be running — the server posts events to it automatically; see `packages/fl/telemetry.py`).

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

Pass `--no-telemetry` to either script to skip posting to the backend entirely. When the run finishes, the server saves the final aggregated model to `packages/models/checkpoints/federated_<strategy>.pt` (same checkpoint format as the baseline, so the two are directly comparable) — override the path with `--out`, or skip saving with `--no-save`.

### 5. Run with real on-chain commitments

Each round can be committed on-chain for real: the coordinator opens/finalises each round, and each client submits its own commitment as its own registered identity (see [Wallet addresses](#wallet-addresses-testnet) below for who needs funding first).

One-time bootstrap per network, after deploying the contract (step 2) and funding the coordinator:

```bash
python -m packages.chain.register_clients --network arbitrum   # or --network base
```

Verify the deployment and bridge work end to end before a full run:

```bash
python -m packages.chain.smoke_test --network arbitrum
```

Then add `--on-chain --chain-network arbitrum` to both the server and every client command from step 4. Each round now costs the coordinator 2 transactions (open + finalise) and each client 1 (submit) — budget testnet ETH accordingly for however many rounds you plan to run.

### 6. Test a transaction

The "Test a Transaction" tab in the frontend lets you hand-build a small transaction context (a target address plus its counterparty edges) and score it against a trained checkpoint, with a plain-language explanation of why it was flagged. It talks to two new backend endpoints:

```
GET  /api/checkpoints   # which checkpoints exist and whether they're scorable
POST /api/score         # {checkpoint, target, edges: [{src, dst, value}, ...]} -> verdict
```

Only checkpoints trained on the L2 simulator's 4-feature schema (`value_in`, `value_out`, `degree_in`, `degree_out`) are scorable this way — an Elliptic-trained checkpoint's 165 features are anonymised, so there's no meaningful way to hand-type a transaction against it. `/api/checkpoints` marks each checkpoint's `scorable` field accordingly; the frontend model picker only lists the scorable ones.

Reasons are computed by `packages/models/reasons.py`, not by the model itself: they're structural heuristics tied directly to the two fraud archetypes `packages/data/l2_simulator.py` injects — a closed multi-hop cycle at a near-uniform inflated value (wash trading), and a high fan-out of counterparties opened and closed within one block (a flash-loan burst). Three sample contexts (normal / wash cycle / flash loan) are built into the tab so you can see a verdict without constructing a graph by hand first.

### 7. Fraud detection screen

The "Fraud Detection" tab checks every wallet in a batch of transactions and shows everything it flags, with the evidence behind each verdict: what the wallet did that block, plain-language reasons, a definition of how that kind of fraud is identified, its counterparties, and — when the answers are known — whether the flag was actually right.

The detector isn't wired into a live rollup node, so transactions reach it one of two ways, chosen at the top of the screen:

- **Practice transactions** — freshly simulated blocks with ground truth attached, so every check is scored for accuracy.
- **Upload a file** — a CSV or JSON of transfers. Required columns: `from`, `to`, `value`. Optional: `block` (groups transfers checked together; defaults to one block), `is_fraud` (1/0, marking the *sending* wallet — include it to get accuracy figures), and `fraud_type` (`wash`/`flash`). The screen offers a downloadable example file with one wash-trading loop and one flash-loan burst hidden in ordinary payments. Features are derived with the same `finalise_features` the simulator uses, with repeat transfers between the same pair in a block summed into one edge. The model learned from transfers of around 1 in value, so rescale files on a very different scale before uploading.

```
GET  /api/fraud/config  # which mode is configured, and whether a scorable checkpoint exists
GET  /api/fraud/model   # the checkpoint behind model mode and its training-time metrics
POST /api/fraud/scan    # {blocks, threshold, seed, mode?, checkpoint?} -> detections + summary
POST /api/fraud/upload  # {format: csv|json, content, threshold, mode?, checkpoint?}
```

Two detector modes, selected by `FRAUD_DETECTION_MODE` in `.env`:

| Mode | What it does |
|---|---|
| `model` | Real inference. Runs the trained checkpoint over **every** address in each simulated block, through the same `forward_node` path (rolling temporal window included) that `packages/fl/task.py` trains and evaluates with. |
| `simulated` | Synthesises verdicts from the generator's ground truth without loading a checkpoint, so the screen works before anything has been trained. Deterministic per seed, and deliberately imperfect — it misses some fraud and raises the occasional false alarm. |

`FRAUD_CHECKPOINT` picks which checkpoint `model` mode uses (default `federated_fedprox`), falling back to any scorable checkpoint on disk. The dropdown on the screen overrides the mode for a single scan, so you can compare the two without restarting the backend.

Both modes emit the same payload, and the summary row scores the sweep against ground truth (caught / missed / false alarms, plus precision, recall and F1). The **Missed** list shows fraudulent addresses that scored *below* the threshold — the ones that would have gone through.

Note the seed: **seed 7 is the stream the federated clients train on** (`packages/fl/client.py` builds its shards from `SimConfig(seed=7)`), so scanning it in `model` mode scores the checkpoint against its own training data. The default seed is deliberately something else, which makes every scan held-out data.

### 8. Evaluating a model for reporting

The metrics stored in a checkpoint come from a validation split of the training stream — roughly 8 blocks, too few for a headline figure. `packages/models/evaluate.py` sweeps a checkpoint over ten independently generated streams it never saw (40 blocks each, ~120,000 wallets), and reports per-stream results, the pooled confusion matrix, pooled precision/recall/F1 and false-alarm rate, per-stream F1 mean ± standard deviation, and recall per fraud archetype:

```bash
python -m packages.models.evaluate --checkpoint baseline --out evaluation-baseline.json
python -m packages.models.evaluate --checkpoint federated_fedprox --out evaluation-fedprox.json
```

Report the per-archetype recall alongside the pooled figures. Flash-loan centres are injected at 50x normal value across 25 counterparties and the model's inputs aren't normalised, so they sit hundreds of times outside the normal feature range and score a saturated probability of 1.0 — near-trivially separable. Wash-trade members look far closer to ordinary traffic per node, so their recall (0.94 for the current `baseline` checkpoint, versus 1.00 for flash) is the more informative measure of what the graph model has learned.

## Wallet addresses (testnet)

Two kinds of identity interact with `FLAggregator`, both testnet-only:

**Coordinator** — derived from `.env`'s `DEPLOYER_PRIVATE_KEY`. Deploys the contract, registers clients, opens/finalises every round. Check the address currently configured with:

```bash
cast wallet address --private-key $DEPLOYER_PRIVATE_KEY
```

**Client dev accounts** — one per simulated FL client, deterministically derived in `packages/chain/dev_accounts.py` from `sha256("l2-fraud-fl-dev-client-{id}")`. Reproducible from the seed alone (no secret involved), so these are always the same addresses on any machine that runs this code:

```bash
python -m packages.chain.dev_accounts --num-clients 3
```

| Role | Address |
|---|---|
| Client 0 | `0xCF57e151Fca68e0e67dbA700Fd76c68EE75083c6` |
| Client 1 | `0xCd0262568B9C2117C74dB6BD6D2ba78677679fe7` |
| Client 2 | `0x87A5025C2fb76a6db4609AC78025f3a25952DA56` |

All 4 addresses (coordinator + 3 clients) need a small amount of testnet ETH before `register_clients.py`, `smoke_test.py`, or any `--on-chain` run — get it from a faucet that doesn't gate on a mainnet balance (e.g. thirdweb's or the Coinbase Developer Platform faucet for the relevant testnet) and send it to each address above. The coordinator burns gas fastest (2 txs/round vs. each client's 1), so top it up more often if it runs dry mid-session.

These keys are derived from public, non-secret seeds and/or live in a local `.env` — never fund any of them with real value, and never reuse this derivation scheme for anything beyond this testnet prototype.

For convenience, `wallets.local.json` (gitignored, generated locally, never committed) holds all 4 addresses and private keys together.

## Build phases

The work is structured as six sprints, each with a distinct deliverable: environment scaffolding, data and graph construction, local model development, federated integration via Flower, L2-native smart contract aggregation, and backend plus frontend visualisation. See `docs/IMPLEMENTATION_GUIDE.md` for the detailed phase-by-phase specification and acceptance criteria.

## Scope

This prototype delivers the design and full implementation (dissertation Objectives 2 and 3), deployed on testnet with a live dashboard. Formal quantitative evaluation and adversarial robustness benchmarking are scoped as future work (Objectives 4 and 5); the telemetry layer is built to feed those later measurement campaigns without rework.
