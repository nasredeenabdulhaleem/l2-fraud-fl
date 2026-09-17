import React from "react";
import { DEFINITIONS, FRAUD_NAMES, VERDICTS, explainReason } from "./copy.js";

function short(hex, n = 10) {
  if (!hex) return "";
  return hex.length > n + 6 ? `${hex.slice(0, n)}...${hex.slice(-4)}` : hex;
}

const money = (x) =>
  x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const DIRECTION = {
  both: "both ways",
  in: "paid this wallet",
  out: "was paid by it",
};

function Definition({ kind }) {
  const def = DEFINITIONS[kind];
  if (!def) return null;
  return (
    <div className="definition">
      <div className="definition-title">
        <span className={`type ${kind}`}>{kind}</span> {def.title}
      </div>
      <p>{def.what}</p>
      <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>How it's spotted:</div>
      <ol>
        {def.signs.map((s) => <li key={s}>{s}</li>)}
      </ol>
    </div>
  );
}

export default function TransactionDetail({ item }) {
  if (!item) {
    return <div className="muted">Pick a wallet from the list to see why it was flagged.</div>;
  }

  const verdict = VERDICTS[item.verdict] || { label: item.verdict, tone: "warn" };
  const truthType = item.ground_truth?.is_fraud ? item.ground_truth.fraud_type : null;

  // Explain whichever kinds of fraud are actually in play for this wallet: the
  // kind it really is (when known) plus any kind a warning sign pointed at.
  const kinds = [...new Set([
    truthType,
    ...item.reasons.map((r) => r.archetype),
  ].filter((k) => DEFINITIONS[k]))];

  return (
    <div>
      <div className="detail-head">
        <span className={`pill ${verdict.tone}`}>{verdict.label}</span>
        {truthType && truthType !== "none" && (
          <span className={`type ${truthType}`}>{FRAUD_NAMES[truthType] || truthType}</span>
        )}
        <span className="muted" style={{ fontSize: 12 }}>batch {item.block}</span>
      </div>

      <div className="mono detail-address">{item.address}</div>
      {item.tx_hash && (
        <div className="muted mono" style={{ fontSize: 11, marginBottom: 14 }}>
          transaction {short(item.tx_hash, 14)}
        </div>
      )}

      <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 10 }}>
        <div style={{ fontSize: 28, fontWeight: 800 }}>{(item.probability * 100).toFixed(1)}%</div>
        <div className="muted" style={{ fontSize: 12 }}>suspicion score</div>
      </div>
      <div className="progress" style={{ margin: "8px 0 6px" }}>
        <span
          style={{
            width: `${item.probability * 100}%`,
            background: item.flagged ? "linear-gradient(90deg, var(--warn), var(--bad))" : undefined,
          }}
        />
      </div>
      <p className="muted" style={{ fontSize: 11, margin: "0 0 18px" }}>
        {item.probability >= 0.9999
          ? "This is the highest score possible. See “Why flash-loan attacks score 100%” in the model section below."
          : "How likely the checker thinks it is that this wallet is involved in fraud."}
      </p>

      <h4 className="detail-label">What this wallet did in this batch</h4>
      <div className="feature-grid">
        <div><span className="muted">Money received</span><strong>{money(item.features.value_in)}</strong></div>
        <div><span className="muted">Money sent</span><strong>{money(item.features.value_out)}</strong></div>
        <div><span className="muted">Wallets that paid it</span><strong>{item.features.degree_in}</strong></div>
        <div><span className="muted">Wallets it paid</span><strong>{item.features.degree_out}</strong></div>
      </div>

      <h4 className="detail-label">Why it looks suspicious</h4>
      <div className="alerts" style={{ marginBottom: 18, maxHeight: "none" }}>
        {item.reasons.map((r, i) => (
          <div className="alert" key={i}>
            {r.archetype && <span className={`type ${r.archetype}`}>{r.archetype}</span>}
            <span style={{ fontSize: 13 }}>{explainReason(r)}</span>
          </div>
        ))}
      </div>

      {kinds.length > 0 && (
        <>
          <h4 className="detail-label">How this kind of fraud is identified</h4>
          {kinds.map((k) => <Definition key={k} kind={k} />)}
        </>
      )}

      <h4 className="detail-label">Wallets it traded with ({item.counterparties.length})</h4>
      <div className="party-table">
        <div className="party-row head">
          <span>Wallet</span><span>Relationship</span><span>Paid to it</span><span>Paid by it</span>
        </div>
        {item.counterparties.slice(0, 12).map((p) => (
          <div className="party-row" key={p.node}>
            <span className="mono">{short(p.address, 8)}</span>
            <span className={`badge ${p.direction === "both" ? "wait" : "done"}`}>
              {DIRECTION[p.direction]}
            </span>
            <span>{money(p.sent_to_target)}</span>
            <span>{money(p.received_from_target)}</span>
          </div>
        ))}
        {item.counterparties.length > 12 && (
          <div className="muted" style={{ fontSize: 11, padding: "6px 2px" }}>
            + {item.counterparties.length - 12} more
          </div>
        )}
      </div>
    </div>
  );
}
