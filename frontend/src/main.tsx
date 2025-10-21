import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Heatmap } from "./Heatmap";

type ModelKey = "bigram" | "t1" | "t2";

type TokenInfo = {
  id: number;
  text: string;
};

type TopItem = {
  token: string;
  id: number;
  logit: number;
  prob: number;
};

type PositionInfo = {
  t: number;
  context_token: TokenInfo;
  next_token: TokenInfo;
  topk: Record<ModelKey, TopItem[] | null>;
  attn: { t1: number[][][]; t2: number[][][] };
  losses: Record<ModelKey, number | null>;
  bigram_available: boolean;
  match_index: number | null;
  match_attention: { t1: number; t2: number } | null;
  skip_trigram: boolean;
};

type AnalysisResponse = {
  tokens: TokenInfo[];
  positions: PositionInfo[];
  device: string;
};

function TokenStrip({
  tokens,
  active,
  onHover,
}: {
  tokens: TokenInfo[];
  active: number | null;
  onHover: (index: number | null) => void;
}) {
  return (
    <div style={{ lineHeight: 1.8, wordBreak: "break-word", userSelect: "none" }}>
      {tokens.map((tok, idx) => {
        const disabled = idx === 0;
        const isActive = active === idx;
        return (
          <span
            key={idx}
            title={disabled ? "bos" : `position ${idx}`}
            onMouseEnter={() => (disabled ? undefined : onHover(idx))}
            onMouseLeave={() => (disabled ? undefined : onHover(null))}
            style={{
              padding: "2px 1px",
              background: isActive ? "rgba(0,160,255,.2)" : disabled ? undefined : "rgba(0,160,255,.05)",
              cursor: disabled ? "default" : "pointer",
              borderBottom: disabled
                ? undefined
                : isActive
                ? "2px solid rgba(0,160,255,.8)"
                : "1px dashed rgba(0,160,255,.35)",
            }}
          >
            {tok.text || "␠"}
          </span>
        );
      })}
    </div>
  );
}

function TopkPanel({ position }: { position: PositionInfo }) {
  const models: { key: ModelKey; title: string }[] = [
    { key: "bigram", title: "Bigram" },
    { key: "t1", title: "Transformer L1" },
    { key: "t2", title: "Transformer L2" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16 }}>
      {models.map((model) => {
        const topkData = position.topk[model.key];
        const loss = position.losses[model.key];
        const isBigram = model.key === "bigram";
        const isUnavailable = isBigram && !position.bigram_available;
        
        return (
          <div 
            key={model.key} 
            style={{ 
              border: "1px solid rgba(0,0,0,.1)", 
              borderRadius: 10, 
              padding: 16,
              opacity: isUnavailable ? 0.5 : 1,
              background: isUnavailable ? "rgba(0,0,0,.05)" : undefined
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 8 }}>
              {model.title}
              {isUnavailable && <span style={{ fontSize: 12, fontWeight: 400, opacity: 0.7 }}> (not in corpus)</span>}
            </div>
            <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 8 }}>
              loss: {loss !== null ? loss.toFixed(2) : "N/A"}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {topkData ? (
                topkData.map((item, idx) => (
                  <div key={item.id} style={{ display: "flex", justifyContent: "space-between", fontFamily: "monospace" }}>
                    <span>{idx + 1}. {item.token || "␠"}</span>
                    <span style={{ opacity: 0.7 }}>p={item.prob.toFixed(4)}</span>
                  </div>
                ))
              ) : (
                <div style={{ fontStyle: "italic", opacity: 0.7, fontSize: 12 }}>
                  No data available
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function AttnPanel({
  label,
  attn,
}: {
  label: string;
  attn: number[][][];
}) {
  return (
    <div>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{label}</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12 }}>
        {attn.map((layer, layerIdx) =>
          layer.map((head, headIdx) => (
            <Heatmap
              key={`${label}-${layerIdx}-${headIdx}`}
              matrix={[head]}
              size={200}
              title={`L${layerIdx} H${headIdx}`}
            />
          )),
        )}
      </div>
    </div>
  );
}

function MatchInfo({ position }: { position: PositionInfo }) {
  if (position.match_index == null) {
    return null;
  }
  const attention = position.match_attention;
  return (
    <div style={{ fontSize: 13, opacity: 0.75 }}>
      repeats token from position {position.match_index}; attention sum →
      {" "}
      t1: {attention ? attention.t1.toFixed(3) : "0.000"}, t2: {attention ? attention.t2.toFixed(3) : "0.000"}
    </div>
  );
}

function App() {
  const [text, setText] = useState("Once upon a time, Alice followed a white rabbit.");
  const [topK, setTopK] = useState(10);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activePosition = useMemo(() => {
    if (!analysis || analysis.positions.length === 0) return null;
    const idx = activeIdx != null ? activeIdx : analysis.positions.length;
    const positionIndex = Math.min(Math.max(idx - 1, 0), analysis.positions.length - 1);
    return analysis.positions[positionIndex];
  }, [analysis, activeIdx]);

  const handleSubmit = async (evt: React.FormEvent) => {
    evt.preventDefault();
    if (!text.trim()) {
      setError("Please enter some text.");
      return;
    }
    setLoading(true);
    setError(null);
    setAnalysis(null);
    setActiveIdx(null);
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, top_k: topK }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail || `server responded ${res.status}`);
      }
      const payload = (await res.json()) as AnalysisResponse;
      setAnalysis(payload);
      setActiveIdx(payload.positions.length ? payload.positions.length : null);
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
      <div style={{ display: "flex", flexDirection: "row", gap: 24, alignItems: "start" }}>
        <pre>{JSON.stringify(analysis, null, 2)}</pre>
        <form onSubmit={handleSubmit} style={{ display: "grid", gap: 12, marginBottom: 24 }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ fontWeight: 600 }}>Text</span>
            <textarea
              value={text}
              onChange={(evt) => setText(evt.target.value)}
              rows={5}
              placeholder="Paste a passage to analyze..."
              style={{ padding: 12, borderRadius: 8, border: "1px solid rgba(0,0,0,.2)", fontFamily: "monospace" }}
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
            {loading ? "Analyzing..." : "Analyze"}
          </button>
          {error && <div style={{ color: "#d42" }}>{error}</div>}
        </form>

      {analysis && (
        <div style={{ display: "grid", gap: 24 }}>
          <div style={{ border: "1px solid rgba(0,0,0,.1)", borderRadius: 10, padding: 16 }}>
            <TokenStrip tokens={analysis.tokens} active={activeIdx} onHover={setActiveIdx} />
            <div style={{ fontSize: 12, opacity: 0.6, marginTop: 6 }}>
              hover a token (after the first) to inspect predictions
            </div>
          </div>

          {activePosition ? (
            <>
              <div style={{ fontSize: 14 }}>
                <strong>Context token:</strong> {activePosition.context_token.text || "␠"}{" "}
                → <strong>next:</strong> {activePosition.next_token.text || "␠"}
              </div>
              <MatchInfo position={activePosition} />
              <TopkPanel position={activePosition} />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                <AttnPanel label="Attention L1" attn={activePosition.attn.t1} />
                <AttnPanel label="Attention L2" attn={activePosition.attn.t2} />
              </div>
            </>
          ) : (
            <div style={{ opacity: 0.7 }}>hover a token to see logits & attention</div>
          )}

          <div style={{ opacity: 0.7, fontSize: 14 }}>
            device: {analysis.device}
          </div>
        </div>
      )}
      </div>

    </div>
  );
}

const container = document.getElementById("root");
if (!container) throw new Error("missing #root");
const root = createRoot(container);
root.render(<App />);
