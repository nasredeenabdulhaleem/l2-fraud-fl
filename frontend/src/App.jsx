import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { useTelemetry } from "./lib/useTelemetry.js";

function short(hex, n = 6) {
  if (!hex) return "";
  return `${hex.slice(0, n)}...${hex.slice(-4)}`;
}

function Stat({ label, value, sub }) {
  return (
    <div className="card stat">
      <div className="value">{value}</div>
      <div className="label">{label}</div>
      {sub}
    </div>
  );
}

export default function App() {
  const { state, connected } = useTelemetry();
  const {
    strategy,
    current_round,
    total_rounds,
    clients,
    round_history,
    fraud_alerts,
  } = state;

  const latest = round_history[round_history.length - 1];
  const progress = total_rounds ? (current_round / total_rounds) * 100 : 0;
  const submitted = clients.filter((c) => c.submitted).length;

  const explorerBase =
    strategy && "https://sepolia.arbiscan.io/tx/"; // Arbitrum Sepolia explorer

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          <div className="logo">FL</div>
          <div>
            <h1>L2 Fraud Detection Monitor</h1>
            <p>Federated learning on Layer 2 rollups, pre-commitment fraud screening</p>
          </div>
        </div>
        <div className="status">
          <span className={`dot ${connected ? "live" : ""}`} />
          {connected ? "Live telemetry" : "Reconnecting..."}
        </div>
      </div>

      {/* Top stat row */}
      <div className="grid">
        <Stat
          label="Aggregation strategy"
          value={<span className="pill accent">{strategy.toUpperCase()}</span>}
        />
        <Stat
          label={`Round ${current_round} of ${total_rounds}`}
          value={`#${current_round}`}
          sub={
            <div className="progress">
              <span style={{ width: `${progress}%` }} />
            </div>
          }
        />
        <Stat
          label="Client submissions this round"
          value={`${submitted}/${clients.length}`}
        />
        <Stat
          label="Global F1 (latest round)"
          value={latest ? latest.f1.toFixed(3) : "--"}
        />
      </div>

      {/* Metric trend + on-chain */}
      <div className="grid two">
        <div className="card">
          <h3>Global model metrics per round</h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={round_history} margin={{ top: 8, right: 16, left: -12, bottom: 0 }}>
              <CartesianGrid stroke="#232d42" strokeDasharray="3 3" />
              <XAxis dataKey="round" stroke="#8a97ad" fontSize={12} />
              <YAxis domain={[0, 1]} stroke="#8a97ad" fontSize={12} />
              <Tooltip
                contentStyle={{
                  background: "#121826",
                  border: "1px solid #232d42",
                  borderRadius: 10,
                  color: "#e6ecf5",
                }}
              />
              <Line type="monotone" dataKey="precision" stroke="#4f8cff" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="recall" stroke="#ffb547" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="f1" stroke="#2ecc9b" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>On-chain finalisation (Aggregator)</h3>
          {latest ? (
            <div>
              <div style={{ marginBottom: 12 }}>
                <div className="muted" style={{ fontSize: 12 }}>Latest finalised round</div>
                <div style={{ fontSize: 22, fontWeight: 800 }}>#{latest.round}</div>
              </div>
              <div style={{ marginBottom: 8 }}>
                <div className="muted" style={{ fontSize: 12 }}>Global model reference (tx)</div>
                <a
                  className="mono"
                  href={`${explorerBase}${latest.tx_hash}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {short(latest.tx_hash, 10)}
                </a>
              </div>
              <div className="muted" style={{ fontSize: 12, lineHeight: 1.6 }}>
                Only a fixed 32-byte commitment is written on chain. The gradient payload
                stays off chain, keeping per-round gas predictable on the rollup.
              </div>
            </div>
          ) : (
            <div className="muted">Waiting for the first finalised round...</div>
          )}
        </div>
      </div>

      {/* Clients + fraud alerts */}
      <div className="grid two">
        <div className="card">
          <h3>Federated client nodes (non-IID shards)</h3>
          {clients.map((c) => (
            <div className="client-row" key={c.id}>
              <div>
                <strong>Node {c.id}</strong>{" "}
                <span className="addr">{c.address}</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span className="muted" style={{ fontSize: 12 }}>
                  fraud rate {(c.fraud_rate * 100).toFixed(1)}%
                </span>
                <span className={`badge ${c.submitted ? "done" : "wait"}`}>
                  {c.submitted ? "submitted" : "training"}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="card">
          <h3>Pre-commitment fraud alerts</h3>
          <div className="alerts">
            {fraud_alerts.length === 0 && (
              <div className="muted">No transactions flagged yet.</div>
            )}
            {[...fraud_alerts].reverse().map((a, i) => (
              <div className="alert" key={`${a.tx}-${i}`}>
                <span className={`type ${a.fraud_type}`}>{a.fraud_type}</span>
                <span className="tx">{short(a.tx, 10)}</span>
                <span className="score">{(a.score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
