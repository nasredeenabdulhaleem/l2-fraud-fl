"""Chain bridge smoke test: open_round -> submit_update -> finalise_round.

A quick way to verify a deployment and the Python bridge work end to end
without running the full FL pipeline. Requires the coordinator and at least
one client dev account (see dev_accounts.py) to already be registered
(register_clients.py) and funded with testnet ETH.

Run:
  python -m packages.chain.smoke_test --network arbitrum
"""

from __future__ import annotations

import argparse
import hashlib
import time

from packages.chain.aggregator_client import AggregatorClient
from packages.chain.dev_accounts import client_account


def _ref(label: str) -> bytes:
    return hashlib.sha256(f"{label}-{time.time()}".encode()).digest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=["arbitrum", "base"], default="arbitrum")
    parser.add_argument("--client-id", type=int, default=0)
    args = parser.parse_args()

    coordinator = AggregatorClient(network=args.network)
    client_key = client_account(args.client_id).key.hex()
    client = AggregatorClient(network=args.network, private_key=client_key)

    print(f"coordinator: {coordinator.acct.address}")
    print(f"client {args.client_id}: {client.acct.address}")

    base_ref = _ref("base-model")
    print(f"\nopen_round(base_ref=0x{base_ref.hex()})")
    r1 = coordinator.open_round(base_ref, window_seconds=3600)
    round_id = coordinator.current_round()
    print(f"  tx={r1.transactionHash.hex()} round_id={round_id}")

    update_ref = _ref("client-update")
    print(f"\nsubmit_update(ref=0x{update_ref.hex()}) as client {args.client_id}")
    r2 = client.submit_update(update_ref)
    print(f"  tx={r2.transactionHash.hex()}")

    stored = coordinator.get_update_ref(round_id, client.acct.address)
    assert stored == update_ref, f"stored ref mismatch: {stored.hex()} != {update_ref.hex()}"
    print(f"  verified on-chain: getUpdateRef matches submitted ref")

    global_ref = _ref("global-model")
    print(f"\nfinalise_round(global_ref=0x{global_ref.hex()})")
    r3 = coordinator.finalise_round(global_ref)
    print(f"  tx={r3.transactionHash.hex()}")

    print("\nsmoke test passed: open -> submit -> finalise round-trip verified on-chain.")


if __name__ == "__main__":
    main()
