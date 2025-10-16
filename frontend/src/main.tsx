import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Heatmap } from "./Heatmap";

type ModelKey = "bigram" | "t1" | "t2";

type TopRow = {
  token: string;
  id: number;
  logit: number;
  prob: number;
};

type HeadMatrix = number[][];
type LayerAttn = HeadMatrix[];
type ModelAttn = LayerAttn[];

type StepMsg = {
  step: number;
  topk: Record<ModelKey, TopRow[]>;
  attn?: {
    t1: ModelAttn;
    t2: ModelAttn;
  };
};

type RunResponse = {
  run_id: string;
  prompt_text: string;
  continuations: Record<ModelKey, string>;
  skip_positions: number[];
  device: string;
  generated_tokens: number;
  requested_tokens: number;
  losses: Record<ModelKey, number>;
};

function Continuation({
  text,
  rows,
  skipPositions = [],
}: {
  text: string;
  rows: StepMsg[];
  skipPositions?: number[];
}) {
  return (
    <div style={{ lineHeight: 1.8, wordBreak: "break-word", userSelect: "none" }}>
      {text.split("").map((ch, i) => {
        const streamed = i < rows.length;
        const isSkip = skipPositions.includes(i);
        return (
          <span
            key={i}
            title={streamed ? `step ${i}` : "pending"}
            style={{
              padding: "2px 1px",
              background: streamed ? "rgba(0,160,255,.08)" : undefined,
              borderBottom: isSkip
                ? "2px solid rgba(255,200,0,.8)"
                : streamed
                ? "1px dashed rgba(0,160,255,.35)"
                : undefined,
            }}
          >
            {ch}
          </span>
        );
      })}
    </div>
  );
}

function ModelTopTable({
  title,
  rows,
  modelKey,
  hoverStep,
  onHover,
}: {
  title: string;
  rows: StepMsg[];
  modelKey: ModelKey;
  hoverStep: number | null;
  onHover: (step: number | null) => void;
}) {
  const data = rows;
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  return (
    <div style={{ flex: 1, minWidth: 280 }}>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{title}</div>
      <div style={{ border: "1px solid rgba(0,0,0,.1)", borderRadius: 8, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "rgba(0,0,0,.04)", textAlign: "left" }}>
              <th style={{ padding: "6px 8px", width: 60 }}>step</th>
              <th style={{ padding: "6px 8px" }}>top 10 tokens</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row) => {
              const active = hoverStep === row.step;
              const isExpanded = expandedStep === row.step;
              return (
                <React.Fragment key={row.step}>
                  <tr
                    onMouseEnter={() => onHover(row.step)}
                    onMouseLeave={() => onHover(null)}
                    onClick={() =>
                      setExpandedStep((prev) => (prev === row.step ? null : row.step))
                    }
                    style={{
                      background: active ? "rgba(0,160,255,.08)" : undefined,
                      cursor: "pointer",
                      borderTop: "1px solid rgba(0,0,0,.07)",
                    }}
                  >
                    <td style={{ padding: "6px 8px", fontVariantNumeric: "tabular-nums" }}>{row.step}</td>
                    <td style={{ padding: "6px 8px" }}>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                        {row.topk[modelKey].map((tok, idx) => (
                          <span
                            key={tok.id}
                            title={`p=${tok.prob.toFixed(4)} logit=${tok.logit.toFixed(3)}`}
                            style={{
                              padding: "2px 6px",
                              borderRadius: 6,
                              background: idx === 0 ? "rgba(0,160,255,.2)" : "rgba(0,0,0,.05)",
                              fontFamily: "monospace",
                              fontSize: 12,
                            }}
                          >
                            {tok.token || "␠"}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                  {isExpanded ? (
                    <tr style={{ background: "rgba(0,0,0,.02)" }}>
                      <td colSpan={2} style={{ padding: "6px 12px" }}>
                        <div style={{ display: "grid", gap: 4 }}>
                          {row.topk[modelKey].map((tok, idx) => (
                            <div
                              key={`${row.step}-${modelKey}-${tok.id}`}
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                fontFamily: "monospace",
                                fontSize: 12,
                                padding: "2px 8px",
                                borderRadius: 4,
                                background:
                                  idx === 0 ? "rgba(0,160,255,.12)" : "rgba(0,0,0,.04)",
                              }}
                            >
                              <span>
                                {tok.token || "␠"} <span style={{ opacity: 0.65 }}>(#{idx + 1})</span>
                              </span>
                              <span style={{ display: "flex", gap: 12 }}>
                                <span>logit {tok.logit.toFixed(3)}</span>
                                <span>p {tok.prob.toFixed(4)}</span>
                              </span>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ) : null}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TopTables({
  rows,
  hoverStep,
  setHoverStep,
}: {
  rows: StepMsg[];
  hoverStep: number | null;
  setHoverStep: (step: number | null) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
      <ModelTopTable
        title="Bigram"
        rows={rows}
        modelKey="bigram"
        hoverStep={hoverStep}
        onHover={setHoverStep}
      />
      <ModelTopTable
        title="Transformer L1"
        rows={rows}
        modelKey="t1"
        hoverStep={hoverStep}
        onHover={setHoverStep}
      />
      <ModelTopTable
        title="Transformer L2"
        rows={rows}
        modelKey="t2"
        hoverStep={hoverStep}
        onHover={setHoverStep}
      />
    </div>
  );
}

function AttnPanel({
  rows,
  hoverStep,
  modelKey,
}: {
  rows: StepMsg[];
  hoverStep: number | null;
  modelKey: "t1" | "t2";
}) {
  const stepIdx = rows.length === 0 ? -1 : hoverStep != null ? hoverStep : rows.length - 1;
  const msg = stepIdx >= 0 ? rows[stepIdx] : undefined;
  const attn = msg?.attn?.[modelKey];
  if (!attn) {
    return <div style={{ opacity: 0.6 }}>no attention yet</div>;
  }

  return (
    <div>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>Attention heatmaps — {modelKey}</div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
          gap: 12,
        }}
      >
        {attn.map((layer, layerIdx) =>
          layer.map((head, headIdx) => (
            <Heatmap
              key={`${layerIdx}-${headIdx}`}
              matrix={head as HeadMatrix}
              size={160}
              title={`L${layerIdx} H${headIdx}`}
            />
          )),
        )}
      </div>
    </div>
  );
}

function App() {
  const [text, setText] = useState("");
  const [genTokens, setGenTokens] = useState(120);
  const [temperature, setTemperature] = useState(1);
  const [topK, setTopK] = useState(50);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<RunResponse | null>(null);
  const [rows, setRows] = useState<StepMsg[]>([]);
  const [hoverStep, setHoverStep] = useState<number | null>(null);
  const [wsStatus, setWsStatus] = useState<"idle" | "open" | "closed" | "error">("idle");

  useEffect(() => {
    if (!run?.run_id) return;

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const hostname = window.location.hostname || "127.0.0.1";
    const currentPort = window.location.port || "";
    const wsPort = currentPort && currentPort !== "8000" ? "8000" : currentPort;
    const authority = wsPort ? `${hostname}:${wsPort}` : hostname;
    const wsUrl = `${protocol}://${authority}/api/stream/${run.run_id}`;
    const ws = new WebSocket(wsUrl);
    setWsStatus("idle");
    setRows([]);

    ws.onopen = () => setWsStatus("open");
    ws.onclose = () => setWsStatus((prev) => (prev === "open" ? "closed" : prev));
    ws.onerror = (event) => {
      console.error("WebSocket stream error", event);
      setWsStatus("error");
    };
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data) as StepMsg;
      setRows((prev) => {
        const next = [...prev];
        next[payload.step] = payload;
        return next;
      });
    };

    return () => {
      ws.close();
    };
  }, [run?.run_id]);

  const handleSubmit = async (evt: React.FormEvent) => {
    evt.preventDefault();
    if (!text.trim()) {
      setError("Please paste some text to condition on.");
      return;
    }
    setError(null);
    setLoading(true);
    setRun(null);
    setRows([]);
    setHoverStep(null);

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          gen_tokens: genTokens,
          temperature,
          top_k: topK,
        }),
      });
      if (!res.ok) {
        throw new Error(`server responded ${res.status}`);
      }
      const payload = (await res.json()) as RunResponse | { error: string };
      if ("error" in payload) {
        setError(payload.error);
      } else {
        setRun(payload);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: "Inter, system-ui, sans-serif", padding: 24, maxWidth: 1200, margin: "0 auto" }}>
      <h1 style={{ marginBottom: 16 }}>Induction Viz</h1>
      <form onSubmit={handleSubmit} style={{ display: "grid", gap: 12, marginBottom: 24 }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontWeight: 600 }}>Prompt text</span>
          <textarea
            value={text}
            onChange={(evt) => setText(evt.target.value)}
            rows={5}
            placeholder="Paste some text to fit the toy models..."
            style={{ padding: 12, borderRadius: 8, border: "1px solid rgba(0,0,0,.2)", fontFamily: "monospace" }}
          />
        </label>
        <div style={{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))" }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span>gen tokens</span>
            <input
              type="number"
              min={1}
              value={genTokens}
              onChange={(evt) => setGenTokens(parseInt(evt.target.value, 10))}
            />
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span>top-k</span>
            <input
              type="number"
              min={1}
              value={topK}
              onChange={(evt) => setTopK(parseInt(evt.target.value, 10) || 1)}
            />
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span>temperature</span>
            <input
              type="number"
              min={0.05}
              step={0.05}
              value={temperature}
              onChange={(evt) => setTemperature(parseFloat(evt.target.value) || 0.05)}
            />
          </label>
        </div>
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "10px 16px",
            borderRadius: 8,
            border: "none",
            background: loading ? "rgba(0,0,0,.2)" : "#0066ff",
            color: "white",
            fontWeight: 600,
            cursor: loading ? "not-allowed" : "pointer",
            width: 160,
          }}
        >
          {loading ? "Running..." : "Run"}
        </button>
        {error && <div style={{ color: "#d42" }}>{error}</div>}
      </form>

      {run && (
        <div style={{ display: "grid", gap: 24 }}>
          <div style={{ display: "grid", gap: 16, gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
            {(["bigram", "t1", "t2"] as ModelKey[]).map((model) => (
              <div
                key={model}
                style={{
                  border: "1px solid rgba(0,0,0,.1)",
                  borderRadius: 10,
                  padding: 16,
                  display: "flex",
                  flexDirection: "column",
                  gap: 12,
                }}
              >
                <div style={{ fontWeight: 600, textTransform: "uppercase", fontSize: 12, letterSpacing: 1 }}>
                  {model}
                </div>
                <Continuation text={run.continuations[model]} rows={rows} skipPositions={run.skip_positions} />
                <div style={{ fontSize: 12, opacity: 0.7 }}>loss: {Number.isFinite(run.losses?.[model]) ? run.losses[model].toFixed(2) : "—"}</div>
              </div>
            ))}
          </div>

          <TopTables rows={rows} hoverStep={hoverStep} setHoverStep={setHoverStep} />

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
            <AttnPanel rows={rows} hoverStep={hoverStep} modelKey="t1" />
            <AttnPanel rows={rows} hoverStep={hoverStep} modelKey="t2" />
          </div>

          <div
            style={{
              fontSize: 14,
              opacity: wsStatus === "error" ? 1 : 0.7,
              color: wsStatus === "error" ? "#d42" : undefined,
            }}
          >
            device: {run.device} — websocket: {wsStatus} — generated tokens: {run.generated_tokens}
          </div>
          {run.generated_tokens < run.requested_tokens && (
            <div style={{ color: "#d42" }}>
              capped by context ({run.generated_tokens} of {run.requested_tokens} requested)
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const container = document.getElementById("root");
if (!container) throw new Error("missing #root");
const root = createRoot(container);
root.render(<App />);
