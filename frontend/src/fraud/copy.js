// Plain-language wording for the fraud screen, kept in one place so the
// definitions shown to a reader stay consistent everywhere they appear.

export const VERDICTS = {
  true_positive: { label: "Fraud caught", tone: "good" },
  false_positive: { label: "False alarm", tone: "warn" },
  false_negative: { label: "Missed fraud", tone: "bad" },
  flagged: { label: "Suspicious", tone: "warn" },
};

export const FRAUD_NAMES = {
  wash: "Wash trading",
  flash: "Flash-loan attack",
  unknown: "Fraud (type not given)",
};

export const DEFINITIONS = {
  wash: {
    title: "Wash trading",
    what:
      "A small group of wallets pass the same money around in a circle — A pays B, B pays C, " +
      "C pays A — to make it look like lots of real trading is happening when it isn't. " +
      "It's used to fake popularity or push a price up.",
    signs: [
      "The money travels in a closed loop through three or more wallets and ends up back where it started.",
      "Each payment around the loop is almost exactly the same amount.",
      "Those amounts are much bigger than a typical payment in the same batch.",
    ],
  },
  flash: {
    title: "Flash-loan attack",
    what:
      "A flash loan lets someone borrow a very large amount and pay it all back within the same " +
      "batch of transactions. The loan itself is allowed, but attackers use those few moments " +
      "of borrowed money to manipulate prices and walk away with a profit.",
    signs: [
      "One wallet deals with an unusually large number of other wallets in a single batch.",
      "With most of them, money goes out and comes straight back — borrow, then repay.",
      "The amounts are many times larger than normal payments.",
    ],
  },
};

// reasons.py phrases its findings for a technical reader; restate them here
// using the evidence it attaches, so the wording matches the rest of the page.
export function explainReason(reason) {
  const ev = reason.evidence || {};
  switch (reason.code) {
    case "closed_cycle": {
      const hops = Math.max(0, (ev.cycle?.length || 1) - 1);
      const legs = ev.leg_values || [];
      const even = legs.length > 0 && Math.max(...legs) / Math.max(Math.min(...legs), 1e-9) < 1.5;
      return (
        `Money went round in a loop through ${hops} wallets and came back to this one` +
        (even ? ", with almost the same amount at every step." : ".")
      );
    }
    case "borrow_repay_burst":
      return (
        `Sent money to, and got it straight back from, ${ev.both_legs} of the ${ev.fanout} ` +
        "wallets it dealt with in this batch — the borrow-then-repay pattern of a flash loan."
      );
    case "high_fanout":
      return `Dealt with an unusually large number of wallets (${ev.fanout}) in a single batch.`;
    case "value_inflation":
      return (
        `Its payments average ${(ev.node_mean / ev.block_mean).toFixed(1)} times the size of a ` +
        "typical payment in this batch."
      );
    case "no_structural_signal":
      return (
        "None of the usual wash-trading or flash-loan warning signs were found. If it was " +
        "flagged, that came from the AI model's own judgement."
      );
    default:
      return reason.summary;
  }
}
