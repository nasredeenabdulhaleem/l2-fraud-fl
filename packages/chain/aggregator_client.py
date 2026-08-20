"""Python bridge to the on-chain FLAggregator.

Lets the Flower server open a round, read submitted commitments, and finalise on
chain through web3.py. Heavy update payloads live off chain; only fixed-size 32-byte
commitments (content hashes) touch the contract, keeping per-transaction gas
predictable on the target optimistic rollups.

Environment (see .env):
  TARGET_NETWORK                arbitrum | base
  ARBITRUM_SEPOLIA_RPC / BASE_SEPOLIA_RPC
  DEPLOYER_PRIVATE_KEY          coordinator key (testnet throwaway only)
  FLAGGREGATOR_ADDRESS_ARBITRUM / FLAGGREGATOR_ADDRESS_BASE
"""

from __future__ import annotations

import hashlib
import os

from web3 import Web3

# Minimal ABI covering the calls and events the coordinator needs.
FLAGGREGATOR_ABI = [
    {"inputs": [], "stateMutability": "nonpayable", "type": "constructor"},
    {"type": "function", "name": "registerClient", "stateMutability": "nonpayable",
     "inputs": [{"name": "client", "type": "address"}], "outputs": []},
    {"type": "function", "name": "openRound", "stateMutability": "nonpayable",
     "inputs": [{"name": "baseModelRef", "type": "bytes32"},
                {"name": "submissionWindow", "type": "uint64"}],
     "outputs": [{"name": "roundId", "type": "uint64"}]},
    {"type": "function", "name": "submitUpdate", "stateMutability": "nonpayable",
     "inputs": [{"name": "ref", "type": "bytes32"}], "outputs": []},
    {"type": "function", "name": "finaliseRound", "stateMutability": "nonpayable",
     "inputs": [{"name": "globalModelRef", "type": "bytes32"}], "outputs": []},
    {"type": "function", "name": "currentRoundId", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint64"}]},
    {"type": "function", "name": "getUpdateRef", "stateMutability": "view",
     "inputs": [{"name": "roundId", "type": "uint64"}, {"name": "client", "type": "address"}],
     "outputs": [{"name": "", "type": "bytes32"}]},
    {"type": "event", "name": "RoundOpened", "anonymous": False,
     "inputs": [{"indexed": True, "name": "roundId", "type": "uint64"},
                {"indexed": False, "name": "baseModelRef", "type": "bytes32"},
                {"indexed": False, "name": "deadline", "type": "uint64"}]},
    {"type": "event", "name": "RoundFinalised", "anonymous": False,
     "inputs": [{"indexed": True, "name": "roundId", "type": "uint64"},
                {"indexed": False, "name": "globalModelRef", "type": "bytes32"},
                {"indexed": False, "name": "submissionCount", "type": "uint32"}]},
]


def model_ref(parameters: list) -> bytes:
    """Deterministic 32-byte commitment to a model parameter list (keccak-style hash).

    The off-chain payload (the actual parameters) is stored separately and referenced
    by this hash. Uses sha256 over the concatenated bytes for portability.
    """
    h = hashlib.sha256()
    for arr in parameters:
        h.update(bytes(memoryview(arr).cast("B")))
    return h.digest()  # 32 bytes


class AggregatorClient:
    def __init__(self, network: str | None = None):
        network = network or os.getenv("TARGET_NETWORK", "arbitrum")
        if network == "arbitrum":
            rpc = os.environ["ARBITRUM_SEPOLIA_RPC"]
            address = os.environ["FLAGGREGATOR_ADDRESS_ARBITRUM"]
        else:
            rpc = os.environ["BASE_SEPOLIA_RPC"]
            address = os.environ["FLAGGREGATOR_ADDRESS_BASE"]

        self.w3 = Web3(Web3.HTTPProvider(rpc))
        self.acct = self.w3.eth.account.from_key(os.environ["DEPLOYER_PRIVATE_KEY"])
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(address), abi=FLAGGREGATOR_ABI
        )

    def _send(self, fn):
        tx = fn.build_transaction({
            "from": self.acct.address,
            "nonce": self.w3.eth.get_transaction_count(self.acct.address),
            "chainId": self.w3.eth.chain_id,
        })
        signed = self.acct.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt

    def open_round(self, base_ref: bytes, window_seconds: int = 3600):
        return self._send(self.contract.functions.openRound(base_ref, window_seconds))

    def finalise_round(self, global_ref: bytes):
        return self._send(self.contract.functions.finaliseRound(global_ref))

    def current_round(self) -> int:
        return int(self.contract.functions.currentRoundId().call())
