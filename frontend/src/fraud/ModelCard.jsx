import React from "react";

const pct = (x) => (x == null ? "--" : `${(x * 100).toFixed(1)}%`);

function Score({ label, value, explain }) {
  return (
    <div className="score-tile">
      <div className="score-value">{value}</div>
      <div className="score-label">{label}</div>
      <div className="muted score-explain">{explain}</div>
    </div>
  );
}

export default function ModelCard({ info, scan }) {
  if (!info) return null;

  if (!info.available) {
    return (
      <div className="card" style={{ marginTop: 16 }}>
        <h3>About the AI model and its accuracy</h3>
        <p className="muted">
          No trained model has been found yet, so only practice mode is available. Train one
          (see the README) and restart the backend.
        </p>
      </div>
    );
  }

  const m = info.metrics || {};
  const measured = scan?.mode === "model" && scan.summary.labelled ? scan.summary : null;

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>About the AI model and its accuracy</h3>

      <p className="model-lede">
        The checker is a graph neural network (GraphSAGE combined with an LSTM) called{" "}
        <code>{info.name}</code>.{" "}
        {info.kind === "federated" ? (
          <>
            It was trained by <strong>{info.clients} separate nodes working together</strong>{" "}
            over {info.rounds} rounds using federated learning ({String(info.strategy).toUpperCase()}):
            each node learned from its own transactions and shared only what it learned, never the
            transactions themselves.
          </>
        ) : (
          <>
            It was trained <strong>in one place on the full practice dataset</strong> — the
            centralised baseline the federated version is compared against.
          </>
        )}{" "}
        It looks at four things about every wallet in a batch — money received, money sent, how
        many wallets paid it, how many it paid — together with who those wallets trade with.
      </p>

      <div className="model-grid">
        <div>
          <h4 className="detail-label">Scores recorded when it was trained</h4>
          <div className="score-row">
            <Score label="Precision" value={pct(m.precision)}
              explain="When it raised an alarm, how often it was right." />
            <Score label="Recall" value={pct(m.recall)}
              explain="Of all the real fraud, how much it caught." />
            <Score label="F1 score" value={pct(m.f1)}
              explain="One number balancing the two above." />
          </div>
          <p className="muted small">
            Measured on the batches held back from its training data (about a fifth of practice
            set {info.training_seed}). That's a small sample, so use the evaluation command below
            for numbers you intend to report.
          </p>
        </div>

        <div>
          <h4 className="detail-label">Scores on the check you just ran</h4>
          {measured ? (
            <>
              <div className="score-row">
                <Score label="Precision" value={pct(measured.precision)}
                  explain={`${measured.true_positives} of ${measured.flagged} alarms were real fraud.`} />
                <Score label="Recall" value={pct(measured.recall)}
                  explain={`Caught ${measured.true_positives} of ${measured.fraud_in_stream} fraudulent wallets.`} />
                <Score label="F1 score" value={pct(measured.f1)}
                  explain={`${measured.false_positives} false alarms among ${measured.scanned.toLocaleString()} wallets.`} />
              </div>
              {scan.source === "simulated" && scan.seed === info.training_seed && (
                <p className="warn-note small">
                  Practice set {info.training_seed} is the data this model learned from, so these
                  scores are flattering. Pick a different practice set number for a fair test.
                </p>
              )}
            </>
          ) : (
            <p className="muted small">
              Run a check with the trained AI model on practice transactions, or on a file that
              includes an <code>is_fraud</code> column, to measure its accuracy here.
            </p>
          )}
        </div>
      </div>

      <details className="explainer">
        <summary>Getting accuracy figures for a report or dissertation</summary>
        <p>
          The scores above come from small samples. For figures you can defend, run the evaluation
          script from the project folder. It tests the model on ten practice sets it has never
          seen (about 120,000 wallets), and prints a results table, the combined confusion matrix,
          precision, recall and F1 with their spread across sets, and how well each type of fraud
          was caught:
        </p>
        <pre className="cmd">python -m packages.models.evaluate --checkpoint {info.name} --out evaluation.json</pre>
        <p>
          Report the <strong>pooled</strong> precision, recall and F1, the per-set F1{" "}
          <strong>mean ± standard deviation</strong>, and the <strong>per-fraud-type</strong>{" "}
          recall. Run it once for the baseline and once for the federated model to compare them
          under identical conditions.
        </p>
      </details>

      <details className="explainer">
        <summary>Why flash-loan attacks score 100%</summary>
        <p>
          In the practice data, a flash-loan wallet moves about <strong>1,250</strong> in one batch,
          while a normal wallet moves around <strong>2 to 6</strong> — hundreds of times less. The
          model is given those amounts as they are, without first rescaling them to a common range,
          so a gap that large pushes its score all the way to the top.
        </p>
        <p>
          So a 100% score means flash-loan attacks in this practice data are{" "}
          <strong>very easy to tell apart</strong>, not that the model is flawless. Wash-trading
          wallets look much more like normal ones and typically score 93–99%, and the model does
          miss a few of them — their catch rate is the better guide to what the model has really
          learned. Real-world flash-loan attacks are usually disguised more carefully than the
          practice data's.
        </p>
      </details>

      <details className="explainer">
        <summary>Limits of uploaded files</summary>
        <p>
          The model learned from practice transactions where a typical payment is about 1. If your
          file uses a very different scale — for example raw token amounts in the billions — its
          scores can be misleading. Rescale amounts so a typical payment is roughly 1 before
          uploading.
        </p>
      </details>
    </div>
  );
}
