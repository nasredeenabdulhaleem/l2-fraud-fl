"""One-time bootstrap: coordinator registers the deterministic client dev accounts.

Run once per network before a live FL run that uses on-chain per-client
submission. Registration is idempotent from the operator's point of view but
NOT from the contract's -- calling this twice for the same client on the same
network reverts with AlreadyRegistered, so this script checks isClient first
and skips already-registered addresses.

Run:
  python -m packages.chain.register_clients --network arbitrum --num-clients 3
"""

from __future__ import annotations

import argparse

from packages.chain.aggregator_client import AggregatorClient
from packages.chain.dev_accounts import client_address


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=["arbitrum", "base"], default="arbitrum")
    parser.add_argument("--num-clients", type=int, default=3)
    args = parser.parse_args()

    coordinator = AggregatorClient(network=args.network)
    print(f"coordinator: {coordinator.acct.address}")

    for i in range(args.num_clients):
        addr = client_address(i)
        already = coordinator.contract.functions.isClient(addr).call()
        if already:
            print(f"client {i} ({addr}): already registered, skipping")
            continue
        receipt = coordinator.register_client(addr)
        print(f"client {i} ({addr}): registered, tx={receipt.transactionHash.hex()}")

    print("\nFund these addresses with testnet ETH before running a live FL round:")
    print(f"  coordinator: {coordinator.acct.address}")
    for i in range(args.num_clients):
        print(f"  client {i}: {client_address(i)}")


if __name__ == "__main__":
    main()
