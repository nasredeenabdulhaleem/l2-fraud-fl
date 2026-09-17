import React, { useEffect, useMemo, useRef, useState } from "react";
import { fetchFraudConfig, fetchModelInfo, runFraudScan, uploadFraudScan } from "./lib/useFraud.js";
import { FRAUD_NAMES, VERDICTS } from "./fraud/copy.js";
import { exampleCsv } from "./fraud/exampleFile.js";
import TransactionDetail from "./fraud/TransactionDetail.jsx";
import ModelCard from "./fraud/ModelCard.jsx";

// Results come back strongest-first; the list replays them batch by batch at a
// steady tick so you watch the batches being checked.
const REVEAL_MS = 90;
const MAX_FILE_BYTES = 5_000_000;

function short(hex, n = 10) {
  if (!hex) return "";
  return hex.length > n + 6 ? `${hex.slice(0, n)}...${hex.slice(-4)}` : hex;
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;

// Mirrors the bounds the backend enforces, so a typo corrects itself instead of
// coming back as an error.
function clamp(value, lo, hi) {
  if (!Number.isFinite(value)) return lo;
  return Math.min(hi, Math.max(lo, value));
}

function Stat({ value, label, sub, tone }) {
  return (
    <div className="card stat">
      <div className="value" style={tone ? { color: `var(--${tone})` } : undefined}>{value}</div>
      <div className="label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export default function FraudScreen() {
  const [config, setConfig] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [source, setSource] = useState("simulate");
  const [mode, setMode] = useState("");
  const [blocks, setBlocks] = useState(12);
  const [threshold, setThreshold] = useState(0.5);
  const [seed, setSeed] = useState(23);
  const [file, setFile] = useState(null);

  const [scan, setScan] = useState(null);
  const [list, setList] = useState("flagged");
  const [selectedId, setSelectedId] = useState(null);
  const [revealed, setRevealed] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fileInput = useRef(null);
  const selectedRef = useRef(null);
  selectedRef.current = selectedId;

  useEffect(() => {
    fetchFraudConfig()
      .then((cfg) => {
        setConfig(cfg);
        setMode(cfg.mode);
        setBlocks(cfg.default_blocks);
        setThreshold(cfg.default_threshold);
        setSeed(cfg.default_seed);
      })
      .catch((err) => setError(err.message));
    fetchModelInfo().then(setModelInfo).catch(() => setModelInfo({ available: false }));
  }, []);

  const ordered = useMemo(() => {
    if (!scan) return [];
    return scan.detections
      .map((d, i) => ({ d, i }))
      .sort((a, b) => a.d.id.split(":")[0] - b.d.id.split(":")[0] || a.i - b.i)
      .map(({ d }) => d);
  }, [scan]);

  useEffect(() => {
    if (!ordered.length) return;
    setRevealed(0);
    const timer = setInterval(() => {
      setRevealed((n) => {
        if (n >= ordered.length) {
          clearInterval(timer);
          return n;
        }
        if (!selectedRef.current) setSelectedId(ordered[0].id);
        return n + 1;
      });
    }, REVEAL_MS);
    return () => clearInterval(timer);
  }, [ordered]);

  function reset() {
    setError(null);
    setScan(null);
    setSelectedId(null);
    setRevealed(0);
    setList("flagged");
  }

  function switchSource(next) {
    setSource(next);
    reset();
  }

  function showList(next) {
    setList(next);
    const first = next === "flagged" ? ordered[0] : (scan?.missed || [])[0];
    setSelectedId(first ? first.id : null);
  }

  function chooseFile(picked) {
    setError(null);
    if (!picked) return;
    const ext = picked.name.split(".").pop().toLowerCase();
    if (!["csv", "json"].includes(ext)) {
      setFile(null);
      setError("Please choose a .csv or .json file.");
      return;
    }
    if (picked.size > MAX_FILE_BYTES) {
      setFile(null);
      setError("That file is larger than 5 MB. Split it into smaller files and check them one at a time.");
      return;
    }
    setFile(picked);
  }

  function downloadExample() {
    const url = URL.createObjectURL(new Blob([exampleCsv()], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "example-transactions.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function submit() {
    reset();
    setLoading(true);
    try {
      const common = { threshold: clamp(threshold, 0, 1), mode };
      if (source === "simulate") {
        setScan(await runFraudScan({
          ...common,
          blocks: clamp(blocks, 1, 40),
          seed: clamp(seed, 0, 10000),
        }));
      } else {
        const format = file.name.split(".").pop().toLowerCase();
        setScan(await uploadFraudScan({ ...common, format, content: await file.text() }));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const summary = scan?.summary;
  const visible = list === "flagged" ? ordered.slice(0, revealed) : scan?.missed || [];
  const selected =
    [...(scan?.detections || []), ...(scan?.missed || [])].find((d) => d.id === selectedId) || null;
  const modelUnavailable = config && !config.checkpoint_available;
  const canRun = !loading && mode && (source === "simulate" || file);

  return (
    <>
      <div className="card intro">
        <h3>Check transactions for fraud</h3>
        <p>
          This page looks for two kinds of fraud — <strong>wash trading</strong> and{" "}
          <strong>flash-loan attacks</strong> — by checking every wallet in a batch of
          transactions and flagging the ones that look suspicious.
        </p>
        <div className="notice">
          <strong>Why you need to provide the transactions:</strong> the checker isn't connected
          to a live blockchain, so it can't watch transactions as they happen. You need to give it
          some to check — either generate practice transactions, or upload a file of your own.
        </div>

        <div className="source-toggle" role="tablist">
          <button
            className={`source ${source === "simulate" ? "on" : ""}`}
            onClick={() => switchSource("simulate")}
          >
            <strong>Use practice transactions</strong>
            <span>Generate realistic transactions with some fraud mixed in, so you can see how well it's caught.</span>
          </button>
          <button
            className={`source ${source === "upload" ? "on" : ""}`}
            onClick={() => switchSource("upload")}
          >
            <strong>Upload my own file</strong>
            <span>Check a CSV or JSON file of transactions you already have.</span>
          </button>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="scan-controls">
          <label className="field">
            <span>Who checks the transactions</span>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="model" disabled={modelUnavailable}>
                The trained AI model{modelUnavailable ? " (not trained yet)" : ""}
              </option>
              <option value="simulated">Practice checker (no AI model needed)</option>
            </select>
            <small>
              {mode === "model"
                ? "The real model scores every wallet."
                : "Imitates a model using the known answers — for trying the page out."}
            </small>
          </label>

          <label className="field">
            <span>How sure before raising an alarm</span>
            <input
              type="number" min="0" max="1" step="0.05" value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              onBlur={() => setThreshold(clamp(threshold, 0, 1))}
            />
            <small>Between 0 and 1. 0.5 means “more likely fraud than not”. Higher means fewer, surer alarms.</small>
          </label>

          {source === "simulate" ? (
            <>
              <label className="field">
                <span>Number of batches</span>
                <input
                  type="number" min="1" max="40" value={blocks}
                  onChange={(e) => setBlocks(Number(e.target.value))}
                  onBlur={() => setBlocks(clamp(blocks, 1, 40))}
                />
                <small>A batch is a group of transactions recorded together (a “block”). Each has 300 wallets.</small>
              </label>
              <label className="field">
                <span>Practice set number</span>
                <input
                  type="number" min="0" max="10000" value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  onBlur={() => setSeed(clamp(seed, 0, 10000))}
                />
                <small>
                  Each number gives a different, repeatable set.
                  {config ? ` Avoid ${config.training_seed} — the model learned from that one.` : ""}
                </small>
              </label>
            </>
          ) : (
            <div className="field upload-field">
              <span>Transactions file</span>
              <div
                className={`dropzone ${file ? "has-file" : ""}`}
                onClick={() => fileInput.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  chooseFile(e.dataTransfer.files[0]);
                }}
              >
                {file ? (
                  <>
                    <strong>{file.name}</strong>
                    <span className="muted">{(file.size / 1024).toFixed(1)} KB · click to choose a different file</span>
                  </>
                ) : (
                  <>
                    <strong>Drop a .csv or .json file here</strong>
                    <span className="muted">or click to browse</span>
                  </>
                )}
              </div>
              <input
                ref={fileInput} type="file" accept=".csv,.json" hidden
                onChange={(e) => chooseFile(e.target.files[0])}
              />
            </div>
          )}

          <button className="btn accent run" onClick={submit} disabled={!canRun}>
            {loading ? "Checking..." : source === "simulate" ? "Run check" : "Check my file"}
          </button>
        </div>

        {source === "upload" && (
          <details className="explainer" open={!file}>
            <summary>How to prepare your file</summary>
            <p>One row per payment. Column names aren't case-sensitive.</p>
            <div className="columns-table">
              <div className="col-row head"><span>Column</span><span>Needed?</span><span>What to put in it</span></div>
              <div className="col-row"><code>from</code><span>Required</span><span>The wallet that sent the money.</span></div>
              <div className="col-row"><code>to</code><span>Required</span><span>The wallet that received it.</span></div>
              <div className="col-row"><code>value</code><span>Required</span><span>How much was sent, as a number.</span></div>
              <div className="col-row"><code>block</code><span>Optional</span><span>Which batch the payment belongs to. Payments in the same batch are checked together. Leave it out to treat the whole file as one batch.</span></div>
              <div className="col-row"><code>is_fraud</code><span>Optional</span><span><strong>1</strong> if the sending wallet is known to be fraudulent, <strong>0</strong> if not. Include it if you want the page to measure how accurate the checker is.</span></div>
              <div className="col-row"><code>fraud_type</code><span>Optional</span><span><code>wash</code> or <code>flash</code>, on rows marked 1.</span></div>
            </div>
            <p className="muted small">
              JSON works too — a list such as{" "}
              <code>{'[{"from": "A", "to": "B", "value": 5}]'}</code>. Up to{" "}
              {config ? config.max_upload_rows.toLocaleString() : "50,000"} payments per file.
            </p>
            <button className="btn ghost" onClick={downloadExample}>Download an example file</button>
            <span className="muted small" style={{ marginLeft: 10 }}>
              Ordinary payments with one wash-trading loop and one flash-loan attack hidden inside.
            </span>
          </details>
        )}

        {error && <div className="alert error-banner">{error}</div>}
      </div>

      {summary && (
        <>
          <div className="grid">
            <Stat
              value={summary.scanned.toLocaleString()}
              label="Wallets checked"
              sub={`in ${summary.blocks} batch${summary.blocks === 1 ? "" : "es"}` +
                (summary.transactions ? ` · ${summary.transactions.toLocaleString()} payments` : "")}
            />
            <Stat
              value={summary.flagged}
              label="Alarms raised"
              sub={summary.labelled
                ? `${summary.fraud_in_stream} wallets were really fraudulent`
                : `scored ${scan.threshold} or higher`}
            />
            {summary.labelled ? (
              <>
                <Stat
                  value={summary.true_positives}
                  tone="good"
                  label="Fraud caught"
                  sub={`${summary.false_negatives} missed · ${summary.false_positives} false alarm${summary.false_positives === 1 ? "" : "s"}`}
                />
                <Stat
                  value={pct(summary.f1)}
                  label="Overall accuracy score (F1)"
                  sub={`Right ${pct(summary.precision)} of the time when it raised an alarm; caught ${pct(summary.recall)} of the real fraud`}
                />
              </>
            ) : (
              <div className="card stat unlabelled">
                <div className="label" style={{ marginTop: 0 }}>Accuracy can't be measured for this file</div>
                <div className="stat-sub">
                  It doesn't say which wallets are really fraudulent, so there's nothing to check the
                  alarms against. Add an <code>is_fraud</code> column to measure accuracy.
                </div>
              </div>
            )}
          </div>

          {summary.labelled && Object.keys(summary.by_type || {}).length > 0 && (
            <div className="by-type">
              {Object.entries(summary.by_type).map(([kind, v]) => (
                <div className="by-type-item" key={kind}>
                  <span className={`type ${kind}`}>{kind}</span>
                  <span>
                    <strong>{FRAUD_NAMES[kind] || kind}:</strong> caught {v.caught} of {v.total} (
                    {pct(v.catch_rate)}) · average score {pct(v.mean_probability)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <div className="grid two">
        <div className="card">
          <div className="list-head">
            <h3 style={{ margin: 0 }}>Results</h3>
            {scan && summary.labelled && (
              <div className="samples" style={{ margin: 0 }}>
                <button className={`btn ghost ${list === "flagged" ? "on" : ""}`} onClick={() => showList("flagged")}>
                  Alarms ({scan.detections.length})
                </button>
                <button className={`btn ghost ${list === "missed" ? "on" : ""}`} onClick={() => showList("missed")}>
                  Missed fraud ({scan.missed.length})
                </button>
              </div>
            )}
          </div>

          {!scan && !loading && (
            <div className="muted">
              {source === "simulate"
                ? "Press “Run check” to generate practice transactions and check them."
                : "Choose a file and press “Check my file”."}
            </div>
          )}
          {loading && <div className="muted">Checking every wallet...</div>}

          {scan && visible.length === 0 && (
            <div className="muted">
              {list === "flagged"
                ? "No wallet scored high enough to raise an alarm."
                : "Nothing slipped through — every fraudulent wallet was caught."}
            </div>
          )}

          <div className="detections">
            {visible.map((d) => {
              const verdict = VERDICTS[d.verdict] || { label: d.verdict, tone: "warn" };
              const kind = d.ground_truth?.is_fraud ? d.ground_truth.fraud_type : null;
              return (
                <button
                  key={d.id}
                  className={`detection ${selectedId === d.id ? "active" : ""}`}
                  onClick={() => setSelectedId(d.id)}
                  title={verdict.label}
                >
                  <span className={`dot-tone ${verdict.tone}`} />
                  <span className="mono">{short(d.address, 10)}</span>
                  <span className="muted block">batch {d.block}</span>
                  {kind && kind !== "none" && <span className={`type ${kind}`}>{kind}</span>}
                  <span className="score">{pct(d.probability)}</span>
                </button>
              );
            })}
          </div>

          {scan && (
            <div className="legend">
              {summary.labelled ? (
                <>
                  <span><i className="dot-tone good" /> fraud caught</span>
                  <span><i className="dot-tone warn" /> false alarm</span>
                  <span><i className="dot-tone bad" /> missed fraud</span>
                </>
              ) : (
                <span><i className="dot-tone warn" /> suspicious — no answers in the file to confirm</span>
              )}
            </div>
          )}

          {scan && list === "flagged" && summary.flagged > summary.detailed && (
            <p className="muted small" style={{ marginBottom: 0 }}>
              Showing the {summary.detailed} highest-scoring of {summary.flagged} alarms. The numbers
              above include all of them.
            </p>
          )}
        </div>

        <div className="card">
          <h3>Wallet details</h3>
          <TransactionDetail item={selected} />
        </div>
      </div>

      <ModelCard info={modelInfo} scan={scan} />
    </>
  );
}
