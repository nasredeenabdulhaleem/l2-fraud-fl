"""Deterministic dev keypairs for the simulated FL clients.

Full per-client on-chain accountability means each simulated client needs its
own identity to call submitUpdate as itself, distinct from the coordinator.
These keys are derived from a fixed, non-secret seed string -- reproducible
across runs, clearly dev/testnet-only, and never meant to hold anything but
small amounts of testnet ETH for gas.
"""

from __future__ import annotations

import hashlib

from eth_account import Account
from eth_account.signers.local import LocalAccount

_SEED_PREFIX = "l2-fraud-fl-dev-client"


def client_account(client_id: int) -> LocalAccount:
    """Return the deterministic dev account for a given client id."""
    seed = hashlib.sha256(f"{_SEED_PREFIX}-{client_id}".encode()).digest()
    return Account.from_key(seed)


def client_address(client_id: int) -> str:
    return client_account(client_id).address


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print deterministic client dev addresses.")
    parser.add_argument("--num-clients", type=int, default=3)
    args = parser.parse_args()

    for i in range(args.num_clients):
        print(f"client {i}: {client_address(i)}")
