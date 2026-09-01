import React, { useEffect, useState } from "react";
import { fetchCheckpoints, scoreTransaction } from "./lib/useScore.js";

// Hand-built contexts for the two archetypes packages/data/l2_simulator.py
// injects, plus a clean baseline -- lets a reviewer see a verdict without
// typing a graph by hand first.
const SAMPLES = {
  normal: {
    label: "Normal traffic",
    target: "A",
    edges: [
      { src: "A", dst: "B", value: 1.0 },
      { src: "C", dst: "A", value: 0.9 },
      { src: "B", dst: "D", value: 1.1 },
    ],
  },
  wash: {
    label: "Wash-trade cycle",
    target: "A",
    edges: [
      { src: "A", dst: "B", value: 5 },
      { src: "B", dst: "C", value: 5 },
      { src: "C", dst: "A", value: 5 },
      { src: "D", dst: "E", value: 0.2 },
      { src: "E", dst: "F", value: 0.2 },
      { src: "F", dst: "D", value: 0.2 },
      { src: "D", dst: "F", value: 0.2 },
    ],
  },
  flash: {
    label: "Flash-loan burst",
    target: "A",
    edges: Array.from({ length: 8 }, (_, i) => [
      { src: "A", dst: `C${i}`, value: 10 },
      { src: `C${i}`, dst: "A", value: 10 },
    ]).flat(),
  },
};

function emptyEdge() {
  return { src: "", dst: "", value: "" };
}

export default function TransactionTester() {
  const [checkpoints, setCheckpoints] = useState([]);
  const [checkpoint, setCheckpoint] = useState("");
  const [target, setTarget] = useState("A");
  const [edges, setEdges] = useState([{ src: "A", dst: "B", value: 1.0 }, emptyEdge()]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchCheckpoints()
      .then((list) => {
        setCheckpoints(list);
        const scorable = list.find((c) => c.scorable);
        if (scorable) setCheckpoint(scorable.name);
      })
      .catch((err) => setError(err.message));
  }, []);

  function loadSample(key) {
    const sample = SAMPLES[key];
    setTarget(sample.target);
    setEdges(sample.edges.map((e) => ({ ...e })));
    setResult(null);
    setError(null);
  }

  function updateEdge(i, field, value) {
    setEdges((prev) => prev.map((e, idx) => (idx === i ? { ...e, [field]: value } : e)));
  }

  function addEdge() {
    setEdges((prev) => [...prev, emptyEdge()]);
  }

  function removeEdge(i) {
    setEdges((prev) => prev.filter((_, idx) => idx !== i));
  }

  async function submit() {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const cleanEdges = edges
        .filter((e) => e.src && e.dst && e.value !== "")
        .map((e) => ({ src: e.src, dst: e.dst, value: Number(e.value) }));
      const res = await scoreTransaction({ checkpoint, target, edges: cleanEdges });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const scorable = checkpoints.filter((c) => c.scorable);
  const unscorable = checkpoints.filter((c) => !c.scorable);

  return (
    <div className="grid two">
      <div className="card">
        <h3>Build a transaction context</h3>

        <div className="samples">
          {Object.entries(SAMPLES).map(([key, s]) => (
            <button key={key} className="btn ghost" onClick={() => loadSample(key)}>
              {s.label}
            </button>
          ))}
        </div>

        <label className="field">
          <span>Model</span>
          <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
            {scorable.length === 0 && <option value="">no scorable checkpoint found</option>}
            {scorable.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.source_or_strategy}
                {c.metrics?.f1 != null ? `, f1=${c.metrics.f1.toFixed(3)}` : ""})
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Target address / node</span>
          <input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="A" />
        </label>

        <div className="edges">
          <div className="edges-head">
            <span>From</span>
            <span>To</span>
            <span>Value</span>
            <span />
          </div>
          {edges.map((edge, i) => (
            <div className="edge-row" key={i}>
              <input value={edge.src} onChange={(e) => updateEdge(i, "src", e.target.value)} placeholder="src" />
              <input value={edge.dst} onChange={(e) => updateEdge(i, "dst", e.target.value)} placeholder="dst" />
              <input
                type="number"
                value={edge.value}
                onChange={(e) => updateEdge(i, "value", e.target.value)}
                placeholder="value"
              />
              <button className="btn icon" onClick={() => removeEdge(i)} title="remove edge">
                &times;
              </button>
            </div>
          ))}
        </div>

        <div className="tester-actions">
          <button className="btn ghost" onClick={addEdge}>
            + add edge
          </button>
          <button className="btn accent" onClick={submit} disabled={loading || !checkpoint}>
            {loading ? "Scoring..." : "Score transaction"}
          </button>
        </div>

        {unscorable.length > 0 && (
          <p className="muted" style={{ fontSize: 11, marginTop: 10 }}>
            {unscorable.map((c) => c.name).join(", ")}{" "}
            {unscorable.length === 1 ? "was" : "were"} trained on Elliptic's anonymised features
            and can't be scored from a hand-built transaction.
          </p>
        )}
      </div>

      <div className="card">
        <h3>Verdict</h3>
        {error && (
          <div className="alert" style={{ borderLeftColor: "var(--bad)" }}>
            {error}
          </div>
        )}
        {!error && !result && (
          <div className="muted">Build a transaction and score it to see a verdict.</div>
        )}
        {result && (
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
              <span className={`pill ${result.is_fraud ? "bad" : "accent"}`}>
                {result.is_fraud ? "FLAGGED" : "clean"}
              </span>
              <div>
                <div style={{ fontSize: 26, fontWeight: 800 }}>
                  {(result.probability * 100).toFixed(1)}%
                </div>
                <div className="muted" style={{ fontSize: 12 }}>
                  fraud probability, target "{result.target}"
                </div>
              </div>
            </div>

            <div className="progress" style={{ marginBottom: 18 }}>
              <span
                style={{
                  width: `${result.probability * 100}%`,
                  background: result.is_fraud
                    ? "linear-gradient(90deg, var(--warn), var(--bad))"
                    : undefined,
                }}
              />
            </div>

            <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
              value_in {result.features.value_in.toFixed(2)} · value_out{" "}
              {result.features.value_out.toFixed(2)} · degree_in {result.features.degree_in} ·
              degree_out {result.features.degree_out}
            </div>

            <div className="alerts">
              {result.reasons.map((r, i) => (
                <div className="alert" key={i}>
                  {r.archetype && <span className={`type ${r.archetype}`}>{r.archetype}</span>}
                  <span>{r.summary}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
