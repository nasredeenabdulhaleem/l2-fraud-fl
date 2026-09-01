"""Structural heuristic explanations for a transaction-scoring verdict.

Ties directly to the two fraud archetypes packages/data/l2_simulator.py injects
(wash trade: a closed cycle through several addresses at an inflated, repeated
value; flash loan: a single address opening and closing a large position across
many counterparties within one block), so a flagged transaction gets an
explanation phrased in the same terms the rest of the system already uses,
rather than an opaque anomaly score. This intentionally does not attempt formal
model-based attribution (GNNExplainer/Captum) -- these are cheap, deterministic
signals computed directly from the submitted edge list.
"""

from __future__ import annotations

_CYCLE_MAX_HOPS = 8
_FANOUT_ELEVATED = 6
_BORROW_REPAY_RATIO = 0.5
_VALUE_INFLATION_RATIO = 2.0


def _find_cycle(edges: list[dict], target: str) -> list[str] | None:
    """DFS for a directed cycle back to target of at least 3 edges.

    A direct 2-edge round trip (target -> X -> target) is the flash-loan
    borrow/repay signature, not a wash cycle -- wash trading specifically
    routes through multiple intermediate addresses to fabricate volume, so
    only cycles with at least one intermediate hop beyond that count here.
    """
    adjacency: dict[str, list[str]] = {}
    for e in edges:
        adjacency.setdefault(str(e["src"]), []).append(str(e["dst"]))

    def dfs(node: str, path: list[str], depth: int) -> list[str] | None:
        if depth > _CYCLE_MAX_HOPS:
            return None
        for nxt in adjacency.get(node, []):
            if nxt == target:
                if depth >= 2:
                    return path + [nxt]
                continue
            if nxt in path:
                continue
            found = dfs(nxt, path + [nxt], depth + 1)
            if found:
                return found
        return None

    return dfs(target, [target], 0)


def _counterparties(edges: list[dict], target: str) -> dict[str, dict[str, bool]]:
    """For each address touching target, which directions were seen."""
    parties: dict[str, dict[str, bool]] = {}
    for e in edges:
        src, dst = str(e["src"]), str(e["dst"])
        if src == target:
            parties.setdefault(dst, {"in": False, "out": False})["out"] = True
        elif dst == target:
            parties.setdefault(src, {"in": False, "out": False})["in"] = True
    return parties


def explain(edges: list[dict], target: str, node_stats: dict) -> list[dict]:
    """Return reason codes for why `target` might be flagged, given its context."""
    reasons: list[dict] = []

    cycle = _find_cycle(edges, target)
    if cycle:
        leg_values = [
            float(e["value"])
            for a, b in zip(cycle, cycle[1:])
            for e in edges
            if str(e["src"]) == a and str(e["dst"]) == b
        ]
        uniform = bool(leg_values) and (max(leg_values) / max(min(leg_values), 1e-9)) < 1.5
        reasons.append({
            "code": "closed_cycle",
            "archetype": "wash",
            "summary": (
                f"Sits on a closed transaction cycle of {len(cycle) - 1} hops"
                + (" at a near-identical value each leg" if uniform else "")
            ),
            "evidence": {"cycle": cycle, "leg_values": leg_values},
        })

    parties = _counterparties(edges, target)
    fanout = len(parties)
    both_legs = sum(1 for p in parties.values() if p["in"] and p["out"])
    if fanout >= _FANOUT_ELEVATED:
        ratio = both_legs / fanout if fanout else 0.0
        if ratio >= _BORROW_REPAY_RATIO:
            reasons.append({
                "code": "borrow_repay_burst",
                "archetype": "flash",
                "summary": (
                    f"Opened and closed positions with {both_legs} of {fanout} "
                    f"counterparties within this block, consistent with a flash-loan burst"
                ),
                "evidence": {"fanout": fanout, "both_legs": both_legs},
            })
        else:
            reasons.append({
                "code": "high_fanout",
                "archetype": "flash",
                "summary": f"Unusually high number of counterparties ({fanout}) in a single block",
                "evidence": {"fanout": fanout},
            })

    block_values = [float(e["value"]) for e in edges]
    block_mean = sum(block_values) / len(block_values) if block_values else 0.0
    node_degree = node_stats["degree_in"] + node_stats["degree_out"]
    node_mean = (node_stats["value_in"] + node_stats["value_out"]) / max(node_degree, 1)
    if block_mean > 0 and node_degree > 0 and node_mean / block_mean >= _VALUE_INFLATION_RATIO:
        reasons.append({
            "code": "value_inflation",
            "archetype": "wash",
            "summary": f"Average transaction value is {node_mean / block_mean:.1f}x the block average",
            "evidence": {"node_mean": node_mean, "block_mean": block_mean},
        })

    if not reasons:
        reasons.append({
            "code": "no_structural_signal",
            "archetype": None,
            "summary": (
                "No wash-trade or flash-loan structural pattern detected; "
                "any flag is driven by the model's learned features alone"
            ),
            "evidence": {},
        })

    return reasons
