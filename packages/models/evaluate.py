"""Held-out evaluation of a trained checkpoint, for reporting.

The metrics stored inside a checkpoint come from a validation split of the
training stream (seed 7) -- around 8 blocks, a thin basis for a headline
number. This sweeps the checkpoint over several independently generated streams
it never saw, pools the confusion matrix across them, and breaks recall down by
fraud archetype, since the two are not equally hard (see the note printed at
the end).

Run:
  python -m packages.models.evaluate --checkpoint baseline
  python -m packages.models.evaluate --checkpoint federated_fedprox --seeds 21 23 31 --out eval.json
"""

from __future__ import annotations

import argparse
import json
import statistics

from packages.models import detector

_DEFAULT_SEEDS = [21, 23, 31, 42, 99, 101, 202, 303, 404, 505]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint name in packages/models/checkpoints (default: FRAUD_CHECKPOINT)")
    parser.add_argument("--seeds", type=int, nargs="+", default=_DEFAULT_SEEDS)
    parser.add_argument("--blocks", type=int, default=40,
                        help="blocks per stream; 40 matches the training stream's size")
    parser.add_argument("--threshold", type=float, default=detector.DEFAULT_THRESHOLD)
    parser.add_argument("--out", default=None, help="also write the full results as JSON")
    args = parser.parse_args()

    seeds = [s for s in args.seeds if s != detector.TRAINING_SEED]
    if len(seeds) != len(args.seeds):
        print(f"note: skipped seed {detector.TRAINING_SEED} -- it's the training stream, "
              f"so it isn't held-out data\n")

    info = detector.model_info(args.checkpoint)
    if not info["available"]:
        raise SystemExit("no scorable checkpoint found in packages/models/checkpoints")

    print(f"checkpoint: {info['name']} ({info['kind']}"
          + (f", {info['strategy']}, {info['rounds']} rounds, {info['clients']} clients"
             if info["kind"] == "federated" else "")
          + ")")
    print(f"stored training-time metrics: {json.dumps(info['metrics'])}")
    print(f"evaluating on {len(seeds)} held-out streams x {args.blocks} blocks, "
          f"threshold {args.threshold}\n")

    header = f"{'seed':>6} {'wallets':>8} {'fraud':>6} {'TP':>4} {'FP':>4} {'FN':>4} " \
             f"{'precision':>9} {'recall':>7} {'F1':>6}"
    print(header)
    print("-" * len(header))

    per_seed, totals = [], {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    kinds: dict[str, dict] = {}
    for seed in seeds:
        s = detector.scan(blocks=args.blocks, threshold=args.threshold, mode="model",
                          checkpoint=info["name"], seed=seed)["summary"]
        per_seed.append({"seed": seed, **s})
        totals["tp"] += s["true_positives"]
        totals["fp"] += s["false_positives"]
        totals["fn"] += s["false_negatives"]
        totals["tn"] += s["true_negatives"]
        for kind, v in s["by_type"].items():
            k = kinds.setdefault(kind, {"total": 0, "caught": 0, "probability_sum": 0.0})
            k["total"] += v["total"]
            k["caught"] += v["caught"]
            k["probability_sum"] += v["mean_probability"] * v["total"]
        print(f"{seed:>6} {s['scanned']:>8} {s['fraud_in_stream']:>6} {s['true_positives']:>4} "
              f"{s['false_positives']:>4} {s['false_negatives']:>4} {s['precision']:>9.4f} "
              f"{s['recall']:>7.4f} {s['f1']:>6.4f}")

    pooled = detector._metrics(totals["tp"], totals["fp"], totals["fn"], totals["tn"])
    f1s = [p["f1"] for p in per_seed]
    f1_sd = statistics.stdev(f1s) if len(f1s) > 1 else 0.0

    print("\npooled confusion matrix (all streams):")
    print(f"                    flagged   not flagged")
    print(f"  actually fraud    {totals['tp']:>7}   {totals['fn']:>11}")
    print(f"  actually clean    {totals['fp']:>7}   {totals['tn']:>11}")
    print(f"\npooled precision {pooled['precision']:.4f}  recall {pooled['recall']:.4f}  "
          f"F1 {pooled['f1']:.4f}  false-alarm rate {pooled['false_alarm_rate']:.6f}")
    print(f"per-stream F1 mean {statistics.mean(f1s):.4f} +/- {f1_sd:.4f} (sd, n={len(f1s)})")

    print("\nby fraud type:")
    by_type = {}
    for kind, k in sorted(kinds.items()):
        by_type[kind] = {
            "total": k["total"],
            "caught": k["caught"],
            "recall": round(k["caught"] / k["total"], 4),
            "mean_probability": round(k["probability_sum"] / k["total"], 6),
        }
        print(f"  {kind:6s} caught {k['caught']}/{k['total']} "
              f"(recall {by_type[kind]['recall']:.4f}), "
              f"mean fraud probability {by_type[kind]['mean_probability']:.6f}")

    print("\nreading these numbers: the simulator injects flash-loan bursts at 50x the "
          "normal transfer value across 25 counterparties, and the model's inputs are "
          "not normalised, so flash-loan centres sit hundreds of times outside the normal "
          "feature range and score a saturated 1.0. Wash-trade members look far closer "
          "to normal traffic per node, so their recall is the more informative measure "
          "of what the graph model has actually learned.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "checkpoint": info,
                "threshold": args.threshold,
                "blocks_per_stream": args.blocks,
                "seeds": seeds,
                "per_seed": per_seed,
                "pooled": {**totals, **pooled},
                "per_stream_f1": {"mean": statistics.mean(f1s), "sd": f1_sd},
                "by_type": by_type,
            }, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
