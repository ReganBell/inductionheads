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

type HeadDelta = {
  magnitude: number;
  actual_token_delta: number;
  top_promoted: Array<{ token: string; id: number; delta: number }>;
  top_suppressed: Array<{ token: string; id: number; delta: number }>;
};

type PositionInfo = {
  t: number;
  context_token: TokenInfo;
  next_token: TokenInfo;
  topk: Record<ModelKey, TopItem[] | null>;
  attn: { t1: number[][][]; t2: number[][][] };
  value_weighted_attn: { t1: number[][][]; t2: number[][][] };
  head_deltas: { t1: Record<string, HeadDelta>; t2: Record<string, HeadDelta> };
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
  t1_layers: number;
  t1_heads: number;
  t2_layers: number;
  t2_heads: number;
};

type AblationResult = {
  with_head: TopItem[];
  without_head: TopItem[];
  delta_positive: TopItem[];
  delta_negative: TopItem[];
};

type CompositionScores = {
  q_composition: number[][];
  k_composition: number[][];
  v_composition: number[][];
};

function TokenStrip({
  tokens,
  active,
  onHover,
  onClick,
  locked,
  attentionData,
  valueWeightedData,
  headDeltasData,
  selectedModel,
  selectedLayer,
  selectedHead,
  highlightMode,
}: {
  tokens: TokenInfo[];
  active: number | null;
  onHover: (index: number | null) => void;
  onClick: (index: number) => void;
  locked: number | null;
  attentionData: { t1: number[][][]; t2: number[][][] } | null;
  valueWeightedData: { t1: number[][][]; t2: number[][][] } | null;
  headDeltasData: { t1: Record<string, HeadDelta>; t2: Record<string, HeadDelta> } | null;
  selectedModel: "t1" | "t2";
  selectedLayer: number;
  selectedHead: number;
  highlightMode: "attention" | "value-weighted" | "delta";
}) {
  // Calculate attention scores for each token using selected head
  const getTokenAttentionScore = (tokenIdx: number): number => {
    if (!attentionData || active === null || active <= 0) return 0;

    const modelData = attentionData[selectedModel];
    if (!modelData || !modelData[selectedLayer] || !modelData[selectedLayer][selectedHead]) {
      return 0;
    }

    const headAttention = modelData[selectedLayer][selectedHead];
    return headAttention[tokenIdx] || 0;
  };

  // Calculate value-weighted attention scores
  const getTokenValueWeightedScore = (tokenIdx: number): number => {
    if (!valueWeightedData || active === null || active <= 0) return 0;

    const modelData = valueWeightedData[selectedModel];
    if (!modelData || !modelData[selectedLayer] || !modelData[selectedLayer][selectedHead]) {
      return 0;
    }

    const headValueWeighted = modelData[selectedLayer][selectedHead];
    return headValueWeighted[tokenIdx] || 0;
  };

  // Calculate delta score for each token using selected head
  const getTokenDeltaScore = (tokenIdx: number, tokenId: number): number => {
    if (!headDeltasData || active === null || active <= 0) return 0;

    const modelData = headDeltasData[selectedModel];
    if (!modelData) return 0;

    const headKey = `L${selectedLayer}H${selectedHead}`;
    const headData = modelData[headKey];
    if (!headData) return 0;

    // Find this token in promoted or suppressed lists
    const promoted = headData.top_promoted.find(t => t.id === tokenId);
    if (promoted) return promoted.delta;

    const suppressed = headData.top_suppressed.find(t => t.id === tokenId);
    if (suppressed) return suppressed.delta;

    return 0;
  };

  // Calculate scores based on mode
  const getTokenScore = (tokenIdx: number, tokenId: number): number => {
    if (highlightMode === "delta") {
      return getTokenDeltaScore(tokenIdx, tokenId);
    } else if (highlightMode === "value-weighted") {
      return getTokenValueWeightedScore(tokenIdx);
    }
    return getTokenAttentionScore(tokenIdx);
  };

  // Calculate max score for normalization
  const maxScore = Math.max(
    ...tokens.map((tok, idx) => Math.abs(getTokenScore(idx, tok.id)))
  );

  return (
    <div style={{ lineHeight: 1.8, wordBreak: "break-word", userSelect: "none" }}>
      {tokens.map((tok, idx) => {
        const disabled = idx === 0;
        const isLocked = locked === idx;
        const isActive = active === idx;
        const score = getTokenScore(idx, tok.id);
        const normalizedScore = maxScore > 0 ? Math.abs(score) / maxScore : 0;

        // Create background color based on mode
        let highlightColor: string;
        if (highlightMode === "delta") {
          // Green for positive delta, red for negative
          if (score > 0) {
            highlightColor = `rgba(46, 207, 139, ${normalizedScore * 0.4})`;
          } else if (score < 0) {
            highlightColor = `rgba(220, 68, 68, ${normalizedScore * 0.4})`;
          } else {
            highlightColor = "rgba(0,0,0,.02)";
          }
        } else {
          // Red for attention
          highlightColor = `rgba(255, 100, 100, ${normalizedScore * 0.3})`;
        }

        const tooltipText = disabled
          ? "bos"
          : highlightMode === "delta"
            ? `position ${idx} (delta: ${score.toFixed(4)}) - click to lock`
            : highlightMode === "value-weighted"
              ? `position ${idx} (value-weighted: ${score.toFixed(3)}) - click to lock`
              : `position ${idx} (attention: ${score.toFixed(3)}) - click to lock`;

        return (
          <span
            key={idx}
            title={tooltipText}
            onMouseEnter={() => (disabled || locked !== null ? undefined : onHover(idx))}
            onMouseLeave={() => (disabled || locked !== null ? undefined : onHover(null))}
            onClick={() => disabled ? undefined : onClick(idx)}
            style={{
              padding: "2px 1px",
              background: isActive
                ? isLocked
                  ? "rgba(46, 207, 139, .25)"  // Green tint when locked
                  : "rgba(0,160,255,.2)"
                : disabled
                  ? undefined
                  : (attentionData || headDeltasData)
                    ? highlightColor
                    : "rgba(0,160,255,.05)",
              cursor: disabled ? "default" : "pointer",
              borderBottom: disabled
                ? undefined
                : isActive
                ? isLocked
                  ? "2px solid #2ECF8B"  // Green border when locked
                  : "2px solid rgba(0,160,255,.8)"
                : "1px dashed rgba(0,160,255,.35)",
              position: "relative",
            }}
          >
            {tok.text || "␠"}
            {isLocked && (
              <span style={{
                fontSize: 10,
                marginLeft: 2,
                opacity: 0.7,
                verticalAlign: "super"
              }}>
                🔒
              </span>
            )}
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
  valueWeightedAttn,
  showValueWeighted,
}: {
  label: string;
  attn: number[][][];
  valueWeightedAttn: number[][][];
  showValueWeighted: boolean;
}) {
  const dataToShow = showValueWeighted ? valueWeightedAttn : attn;
  const suffix = showValueWeighted ? " (value-weighted)" : "";
  
  return (
    <div>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{label}{suffix}</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12 }}>
        {dataToShow.map((layer, layerIdx) =>
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

function HeadInfluencePanel({
  position,
  selectedModel,
  selectedLayer,
  selectedHead,
  onHeadClick,
}: {
  position: PositionInfo;
  selectedModel: "t1" | "t2";
  selectedLayer: number;
  selectedHead: number;
  onHeadClick?: (layer: number, head: number) => void;
}) {
  const deltas = position.head_deltas[selectedModel];

  // Convert to array and sort by magnitude
  const sortedHeads = Object.entries(deltas)
    .map(([headKey, headData]) => ({ headKey, ...headData }))
    .sort((a, b) => b.magnitude - a.magnitude);

  const topHeads = sortedHeads.slice(0, 8);

  const selectedHeadKey = `L${selectedLayer}H${selectedHead}`;
  const selectedHeadData = deltas[selectedHeadKey];

  return (
    <div style={{
      border: "1px solid rgba(0,0,0,.1)",
      borderRadius: 10,
      padding: 16,
      background: "#FCFCFC",
    }}>
      <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 14 }}>
        Head Influence Analysis ({selectedModel.toUpperCase()})
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Left: Top heads by magnitude */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, opacity: 0.7 }}>
            Most Influential Heads (by magnitude)
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {topHeads.map(({ headKey, magnitude, actual_token_delta }) => {
              const isSelected = headKey === selectedHeadKey;
              return (
                <div
                  key={headKey}
                  style={{
                    fontSize: 12,
                    fontFamily: "monospace",
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "4px 6px",
                    borderRadius: 4,
                    background: isSelected ? "rgba(0,160,255,.1)" : undefined,
                    border: isSelected ? "1px solid rgba(0,160,255,.3)" : "1px solid transparent",
                    cursor: onHeadClick ? "pointer" : "default",
                  }}
                  onClick={() => {
                    if (onHeadClick) {
                      const match = headKey.match(/L(\d+)H(\d+)/);
                      if (match) {
                        onHeadClick(parseInt(match[1]), parseInt(match[2]));
                      }
                    }
                  }}
                >
                  <span>{headKey}</span>
                  <span style={{ opacity: 0.7 }}>mag: {magnitude.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Selected head details */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, opacity: 0.7 }}>
            Selected Head: {selectedHeadKey}
          </div>
          {selectedHeadData ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ fontSize: 11, opacity: 0.7 }}>
                Magnitude: {selectedHeadData.magnitude.toFixed(2)} |
                Actual token Δ: {selectedHeadData.actual_token_delta.toFixed(4)}
              </div>

              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: "#2ECF8B", marginBottom: 4 }}>
                  Top Promoted:
                </div>
                {selectedHeadData.top_promoted.length > 0 ? (
                  selectedHeadData.top_promoted.slice(0, 5).map((item, i) => (
                    <div key={i} style={{ fontSize: 11, fontFamily: "monospace", display: "flex", justifyContent: "space-between" }}>
                      <span>{item.token || "␠"}</span>
                      <span style={{ color: "#2ECF8B" }}>+{item.delta.toFixed(3)}</span>
                    </div>
                  ))
                ) : (
                  <div style={{ fontSize: 10, opacity: 0.5, fontStyle: "italic" }}>None</div>
                )}
              </div>

              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: "#d42", marginBottom: 4 }}>
                  Top Suppressed:
                </div>
                {selectedHeadData.top_suppressed.length > 0 ? (
                  selectedHeadData.top_suppressed.slice(0, 5).map((item, i) => (
                    <div key={i} style={{ fontSize: 11, fontFamily: "monospace", display: "flex", justifyContent: "space-between" }}>
                      <span>{item.token || "␠"}</span>
                      <span style={{ color: "#d42" }}>{item.delta.toFixed(3)}</span>
                    </div>
                  ))
                ) : (
                  <div style={{ fontSize: 10, opacity: 0.5, fontStyle: "italic" }}>None</div>
                )}
              </div>
            </div>
          ) : (
            <div style={{ fontSize: 11, opacity: 0.6, fontStyle: "italic" }}>No data</div>
          )}
        </div>
      </div>
    </div>
  );
}

function AblationPanel({
  ablation,
  onClose,
}: {
  ablation: AblationResult;
  onClose: () => void;
}) {
  const panels: { key: keyof AblationResult; title: string; description: string }[] = [
    { key: "with_head", title: "With Head", description: "Top predictions with the head active" },
    { key: "without_head", title: "Without Head", description: "Top predictions with the head ablated" },
    { key: "delta_positive", title: "Most Helped (+Δ)", description: "Tokens most promoted by this head" },
    { key: "delta_negative", title: "Most Hurt (−Δ)", description: "Tokens most suppressed by this head" },
  ];

  return (
    <div style={{
      border: "2px solid #2ECF8B",
      borderRadius: 10,
      padding: 20,
      background: "#FCFCFC",
      position: "relative",
    }}>
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          background: "rgba(0,0,0,.05)",
          border: "1px solid rgba(0,0,0,.1)",
          borderRadius: 4,
          padding: "4px 8px",
          cursor: "pointer",
          fontSize: 12,
        }}
      >
        Close
      </button>

      <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 4 }}>
        Head Ablation Analysis
      </div>
      <div style={{ fontSize: 13, opacity: 0.6, fontStyle: "italic", marginBottom: 16 }}>
        Misleading, Probably Useless
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16 }}>
        {panels.map((panel) => {
          const data = ablation[panel.key];
          return (
            <div
              key={panel.key}
              style={{
                border: "1px solid rgba(0,0,0,.1)",
                borderRadius: 8,
                padding: 14,
                background: "white",
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 4, fontSize: 14 }}>
                {panel.title}
              </div>
              <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 10 }}>
                {panel.description}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {data.map((item, idx) => (
                  <div
                    key={item.id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontFamily: "monospace",
                      fontSize: 13,
                    }}
                  >
                    <span>{idx + 1}. {item.token || "␠"}</span>
                    <span style={{ opacity: 0.7 }}>
                      {panel.key.startsWith("delta")
                        ? `Δ${item.logit >= 0 ? "+" : ""}${item.logit.toFixed(2)}`
                        : `p=${item.prob.toFixed(4)}`
                      }
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AttentionHeadSelector({
  analysis,
  selectedModel,
  selectedLayer,
  selectedHead,
  onModelChange,
  onLayerChange,
  onHeadChange,
  compositionScores,
  showComposition,
  onToggleComposition,
}: {
  analysis: AnalysisResponse;
  selectedModel: "t1" | "t2";
  selectedLayer: number;
  selectedHead: number;
  onModelChange: (model: "t1" | "t2") => void;
  onLayerChange: (layer: number) => void;
  onHeadChange: (head: number) => void;
  compositionScores: CompositionScores | null;
  showComposition: boolean;
  onToggleComposition: (show: boolean) => void;
}) {
  const layers = selectedModel === "t1" ? analysis.t1_layers : analysis.t2_layers;
  const heads = selectedModel === "t1" ? analysis.t1_heads : analysis.t2_heads;

  const headRadius = 24;
  const layerSpacing = 120;
  const headSpacing = 60;

  // Calculate positions for heads (vertical layout: Layer 0 top, Layer 1 bottom)
  const getHeadPosition = (layer: number, head: number) => {
    const totalWidth = (heads - 1) * headSpacing;
    const x = head * headSpacing - totalWidth / 2 + 240;
    const y = layer * layerSpacing + 60;
    return { x, y };
  };

  // Get composition scores between two heads
  const getKCompositionScore = (l1Head: number, l0Head: number): number => {
    if (!compositionScores || selectedModel === "t1") return 0;
    return compositionScores.k_composition[l1Head]?.[l0Head] || 0;
  };

  const getQCompositionScore = (l1Head: number, l0Head: number): number => {
    if (!compositionScores || selectedModel === "t1") return 0;
    return compositionScores.q_composition[l1Head]?.[l0Head] || 0;
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 16 }}>
      <div style={{ fontWeight: 600, fontSize: 14 }}>Model Architecture</div>

      {/* Model Selection */}
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input
            type="radio"
            checked={selectedModel === "t1"}
            onChange={() => onModelChange("t1")}
          />
          <span style={{ fontWeight: 500 }}>1-Layer Attention Only</span>
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input
            type="radio"
            checked={selectedModel === "t2"}
            onChange={() => onModelChange("t2")}
          />
          <span style={{ fontWeight: 500 }}>2-Layer Attention Only</span>
        </label>
        {selectedModel === "t2" && (
          <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", marginLeft: 12, fontSize: 13 }}>
            <input
              type="checkbox"
              checked={showComposition}
              onChange={(e) => onToggleComposition(e.target.checked)}
            />
            <span>Show Head Composition</span>
          </label>
        )}
      </div>

      {/* Visual Architecture */}
      <div style={{ position: "relative", width: "100%", height: selectedModel === "t2" ? 260 : 140, background: "#FCFCFC", border: "1px solid rgba(0,0,0,.1)", borderRadius: 10, overflow: "hidden" }}>
        <svg style={{ width: "100%", height: "100%", position: "absolute", top: 0, left: 0 }}>
          {/* Draw composition connections for 2-layer model */}
          {selectedModel === "t2" && showComposition && compositionScores && (
            <>
              {/* K-Composition (green) - strongest signal */}
              {Array.from({ length: heads }, (_, l1Head) =>
                Array.from({ length: heads }, (_, l0Head) => {
                  const score = getKCompositionScore(l1Head, l0Head);
                  if (score <= 0.001) return null; // Only draw significant positive composition

                  const l0Pos = getHeadPosition(0, l0Head);
                  const l1Pos = getHeadPosition(1, l1Head);

                  // Exponential scaling to emphasize strong connections
                  const normalizedScore = Math.max(0, score / 0.05); // Normalize assuming max ~0.05
                  const intensity = Math.pow(normalizedScore, 2); // Exponential emphasis
                  const opacity = Math.min(intensity * 0.9, 0.9);
                  const strokeWidth = Math.max(1, intensity * 8);

                  return (
                    <line
                      key={`k-conn-${l1Head}-${l0Head}`}
                      x1={l0Pos.x}
                      y1={l0Pos.y + headRadius}
                      x2={l1Pos.x}
                      y2={l1Pos.y - headRadius}
                      stroke="#2ECF8B"
                      strokeWidth={strokeWidth}
                      opacity={opacity}
                      strokeLinecap="round"
                    />
                  );
                })
              )}

              {/* Q-Composition (blue) - weaker signal */}
              {Array.from({ length: heads }, (_, l1Head) =>
                Array.from({ length: heads }, (_, l0Head) => {
                  const score = getQCompositionScore(l1Head, l0Head);
                  if (score <= 0.001) return null; // Only draw significant positive composition

                  const l0Pos = getHeadPosition(0, l0Head);
                  const l1Pos = getHeadPosition(1, l1Head);

                  // Exponential scaling
                  const normalizedScore = Math.max(0, score / 0.05);
                  const intensity = Math.pow(normalizedScore, 2);
                  const opacity = Math.min(intensity * 0.7, 0.7);
                  const strokeWidth = Math.max(0.5, intensity * 6);

                  return (
                    <line
                      key={`q-conn-${l1Head}-${l0Head}`}
                      x1={l0Pos.x}
                      y1={l0Pos.y + headRadius}
                      x2={l1Pos.x}
                      y2={l1Pos.y - headRadius}
                      stroke="#0066ff"
                      strokeWidth={strokeWidth}
                      opacity={opacity}
                      strokeLinecap="round"
                    />
                  );
                })
              )}
            </>
          )}
        </svg>

        {/* Draw heads */}
        {Array.from({ length: layers }, (_, layer) => (
          <div key={`layer-${layer}`} style={{ position: "absolute", left: 0, top: 0 }}>
            {/* Layer label */}
            <div style={{
              position: "absolute",
              left: 20,
              top: layer * layerSpacing + 60 - 10,
              fontSize: 13,
              fontWeight: 600,
              opacity: 0.7,
            }}>
              Layer {layer}
            </div>

            {/* Heads for this layer */}
            {Array.from({ length: heads }, (_, head) => {
              const pos = getHeadPosition(layer, head);
              const isSelected = selectedLayer === layer && selectedHead === head;

              return (
                <div
                  key={`head-${layer}-${head}`}
                  onClick={() => {
                    onLayerChange(layer);
                    onHeadChange(head);
                  }}
                  style={{
                    position: "absolute",
                    left: pos.x - headRadius,
                    top: pos.y - headRadius,
                    width: headRadius * 2,
                    height: headRadius * 2,
                    borderRadius: "50%",
                    border: isSelected ? "3px solid #0066ff" : "2px solid rgba(0,0,0,.2)",
                    background: isSelected ? "#0066ff" : "white",
                    color: isSelected ? "white" : "#0E1111",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 11,
                    fontWeight: 600,
                    cursor: "pointer",
                    transition: "all 0.2s",
                    boxShadow: isSelected ? "0 4px 12px rgba(0,102,255,.3)" : "0 2px 4px rgba(0,0,0,.1)",
                  }}
                  title={`L${layer}H${head}`}
                >
                  {head}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <div style={{ fontSize: 12, opacity: 0.7 }}>
        Selected: {selectedModel.toUpperCase()} L{selectedLayer}H{selectedHead}
        {selectedModel === "t2" && showComposition && compositionScores && " • Green = K-composition, Blue = Q-composition"}
      </div>
    </div>
  );
}

function CompositionScoresPanel({
  compositionScores,
  onClose,
}: {
  compositionScores: CompositionScores;
  onClose: () => void;
}) {
  const { q_composition, k_composition, v_composition } = compositionScores;

  const renderHeatmap = (data: number[][], title: string, description: string) => {
    if (!data || data.length === 0 || data[0].length === 0) return null;

    const nHeads = data.length;
    const maxVal = Math.max(...data.flat().map(Math.abs));

    return (
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>{title}</div>
        <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 10 }}>{description}</div>
        <div style={{ display: "inline-grid", gridTemplateColumns: `40px repeat(${nHeads}, 40px)`, gap: 2 }}>
          {/* Header row - layer 0 heads */}
          <div></div>
          {Array.from({ length: nHeads }, (_, i) => (
            <div key={i} style={{ fontSize: 10, textAlign: "center", opacity: 0.6 }}>
              L0H{i}
            </div>
          ))}

          {/* Rows for each layer 1 head */}
          {data.map((row, l1Head) => (
            <React.Fragment key={l1Head}>
              <div style={{ fontSize: 10, display: "flex", alignItems: "center", opacity: 0.6 }}>
                L1H{l1Head}
              </div>
              {row.map((value, l0Head) => {
                const intensity = maxVal > 0 ? Math.abs(value) / maxVal : 0;
                const color = value >= 0
                  ? `rgba(46, 207, 139, ${intensity})`  // Green for positive
                  : `rgba(255, 68, 68, ${intensity})`;   // Red for negative

                return (
                  <div
                    key={l0Head}
                    style={{
                      width: 40,
                      height: 40,
                      background: color,
                      border: "1px solid rgba(0,0,0,.1)",
                      borderRadius: 4,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 9,
                      fontFamily: "monospace",
                    }}
                    title={`L1H${l1Head} ← L0H${l0Head}: ${value.toFixed(3)}`}
                  >
                    {value.toFixed(2)}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div style={{
      border: "2px solid #0066ff",
      borderRadius: 10,
      padding: 20,
      background: "#FCFCFC",
      position: "relative",
    }}>
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          background: "rgba(0,0,0,.05)",
          border: "1px solid rgba(0,0,0,.1)",
          borderRadius: 4,
          padding: "4px 8px",
          cursor: "pointer",
          fontSize: 12,
        }}
      >
        Close
      </button>

      <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 16 }}>
        Head Composition Scores (Layer 1 ← Layer 0)
      </div>

      <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 16 }}>
        Shows how much information flows between attention heads through Q-, K-, and V-composition pathways.
        Higher values = stronger composition. Computed from model weights (input-independent).
      </div>

      <div style={{ display: "grid", gap: 24 }}>
        {renderHeatmap(k_composition, "K-Composition", "How much does L1 key vector read from L0 output?")}
        {renderHeatmap(q_composition, "Q-Composition", "How much does L1 query vector read from L0 output?")}
        {renderHeatmap(v_composition, "V-Composition", "How much does L1 value vector read from L0 output?")}
      </div>
    </div>
  );
}

function AblateHeadButton({
  onClick,
  loading,
  disabled,
  position,
  model,
  layer,
  head,
}: {
  onClick: () => void;
  loading: boolean;
  disabled: boolean;
  position?: number;
  model: string;
  layer: number;
  head: number;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <button
        onClick={onClick}
        disabled={loading || disabled}
        style={{
          padding: "10px 16px",
          borderRadius: 6,
          border: "1px solid #2ECF8B",
          background: loading || disabled ? "rgba(0,0,0,.1)" : "#2ECF8B",
          color: loading || disabled ? "rgba(0,0,0,.4)" : "white",
          fontWeight: 600,
          cursor: loading || disabled ? "not-allowed" : "pointer",
          fontSize: 14,
        }}
      >
        {loading ? "Analyzing..." : "Ablate Selected Head"}
      </button>
      {position !== undefined && (
        <div style={{ fontSize: 12, opacity: 0.7 }}>
          Will analyze {model.toUpperCase()} L{layer}H{head} at position {position}
        </div>
      )}
    </div>
  );
}

function App() {
  const [text, setText] = useState("^Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large moustache. Mrs Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbours. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn't think they could bear it if anyone found out about the Potters. Mrs Potter was Mrs Dursley's sister, but they hadn't met for several years; in fact, Mrs Dursley pretended she didn't have a sister, because her sister and her good- for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbours would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn't want Dudley mixing with a child like that.");
  const [topK, setTopK] = useState(10);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [lockedIdx, setLockedIdx] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showValueWeighted, setShowValueWeighted] = useState(false);
  const [selectedModel, setSelectedModel] = useState<"t1" | "t2">("t1");
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [ablationResult, setAblationResult] = useState<AblationResult | null>(null);
  const [ablationLoading, setAblationLoading] = useState(false);
  const [highlightMode, setHighlightMode] = useState<"attention" | "value-weighted" | "delta">("attention");
  const [computeAblations, setComputeAblations] = useState(false);
  const [compositionScores, setCompositionScores] = useState<CompositionScores | null>(null);
  const [compositionLoading, setCompositionLoading] = useState(false);
  const [showTextEditor, setShowTextEditor] = useState(false);
  const [showComposition, setShowComposition] = useState(true);

  // Auto-load on mount
  React.useEffect(() => {
    if (!analysis && !loading) {
      handleSubmit(new Event('submit') as any);
    }
  }, []);

  // Auto-load composition scores when switching to T2 or when toggle is turned on
  React.useEffect(() => {
    if (selectedModel === "t2" && showComposition && !compositionScores && !compositionLoading) {
      handleLoadCompositionScores();
    }
  }, [selectedModel, showComposition]);

  const activePosition = useMemo(() => {
    if (!analysis || analysis.positions.length === 0) return null;
    // Use locked index if available, otherwise use active/hover index
    const idx = lockedIdx != null ? lockedIdx : (activeIdx != null ? activeIdx : analysis.positions.length);
    const positionIndex = Math.min(Math.max(idx - 1, 0), analysis.positions.length - 1);
    return analysis.positions[positionIndex];
  }, [analysis, activeIdx, lockedIdx]);

  const handleTokenClick = (idx: number) => {
    // Toggle lock: if clicking the same token, unlock; otherwise lock to new token
    if (lockedIdx === idx) {
      setLockedIdx(null);
    } else {
      setLockedIdx(idx);
      setActiveIdx(idx);
    }
  };

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
        body: JSON.stringify({ text, top_k: topK, compute_ablations: computeAblations }),
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

  const handleAblateHead = async () => {
    // Use the active position from the analysis, not the hover state
    const position = activePosition?.t;
    if (!text.trim() || position === undefined) {
      setError("Please analyze text first.");
      return;
    }
    setAblationLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/ablate-head", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          position,
          model_name: selectedModel,
          layer: selectedLayer,
          head: selectedHead,
          top_k: topK,
        }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail || `server responded ${res.status}`);
      }
      const payload = (await res.json()) as AblationResult;
      setAblationResult(payload);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      setAblationLoading(false);
    }
  };

  const handleLoadCompositionScores = async () => {
    if (selectedModel === "t1") {
      setError("Composition scores require a 2-layer model. Please select T2.");
      return;
    }
    setCompositionLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/composition-scores?model_name=${selectedModel}`);
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail || `server responded ${res.status}`);
      }
      const payload = (await res.json()) as CompositionScores;
      setCompositionScores(payload);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      setCompositionLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: "Inter, system-ui, sans-serif", padding: 24, maxWidth: 1200, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ margin: 0 }}>Induction Heads 🎉</h1>
        <button
          onClick={() => setShowTextEditor(!showTextEditor)}
          style={{
            padding: "8px 14px",
            borderRadius: 6,
            border: "1px solid rgba(0,0,0,.2)",
            background: showTextEditor ? "rgba(0,0,0,.05)" : "white",
            cursor: "pointer",
            fontSize: 14,
            fontWeight: 500,
          }}
        >
          {showTextEditor ? "Hide Editor" : "Edit Text"}
        </button>
      </div>

      {showTextEditor && (
        <form onSubmit={handleSubmit} style={{ display: "grid", gap: 12, marginBottom: 24, padding: 16, border: "1px solid rgba(0,0,0,.1)", borderRadius: 10, background: "#FCFCFC" }}>
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
              // min={1}
              value={topK}
              onChange={(evt) => setTopK(parseInt(evt.target.value, 10) || 1)}
            />
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={computeAblations}
              onChange={(evt) => setComputeAblations(evt.target.checked)}
            />
            <span>Compute ablations (slow!)</span>
          </label>
          {computeAblations && (
            <div style={{ fontSize: 12, opacity: 0.7, fontStyle: "italic" }}>
              ⚠️ This will run many forward passes (layers × heads per token).
              Enables "Head Delta" highlighting mode.
            </div>
          )}
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
      )}

      {analysis && (
        <div style={{ display: "grid", gap: 24 }}>
          <AttentionHeadSelector
            analysis={analysis}
            selectedModel={selectedModel}
            selectedLayer={selectedLayer}
            selectedHead={selectedHead}
            onModelChange={setSelectedModel}
            onLayerChange={setSelectedLayer}
            onHeadChange={setSelectedHead}
            compositionScores={compositionScores}
            showComposition={showComposition}
            onToggleComposition={setShowComposition}
          />

          <AblateHeadButton
            onClick={handleAblateHead}
            loading={ablationLoading}
            disabled={!activePosition}
            position={activePosition?.t}
            model={selectedModel}
            layer={selectedLayer}
            head={selectedHead}
          />

          {ablationResult && (
            <AblationPanel
              ablation={ablationResult}
              onClose={() => setAblationResult(null)}
            />
          )}

          <div style={{ border: "1px solid rgba(0,0,0,.1)", borderRadius: 10, padding: 16 }}>
            <TokenStrip
              tokens={analysis.tokens}
              active={lockedIdx != null ? lockedIdx : activeIdx}
              onHover={setActiveIdx}
              onClick={handleTokenClick}
              locked={lockedIdx}
              attentionData={activePosition?.attn || null}
              valueWeightedData={activePosition?.value_weighted_attn || null}
              headDeltasData={activePosition?.head_deltas || null}
              selectedModel={selectedModel}
              selectedLayer={selectedLayer}
              selectedHead={selectedHead}
              highlightMode={highlightMode}
            />
            <div style={{ fontSize: 12, opacity: 0.6, marginTop: 6, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>
                Hover a token (after the first) to inspect predictions. Click to lock selection.
                {highlightMode === "attention" && " Red = attention weight."}
                {highlightMode === "value-weighted" && " Red = attention × ||value|| (mechanistic info flow)."}
                {highlightMode === "delta" && " Green = promoted by head, Red = suppressed (causal effect)."}
              </span>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, cursor: "pointer" }}>
                <span>Highlight:</span>
                <select
                  value={highlightMode}
                  onChange={(e) => {
                    const newMode = e.target.value as "attention" | "value-weighted" | "delta";
                    // Don't allow delta mode if no ablations were computed
                    if (newMode === "delta" && (!activePosition?.head_deltas || Object.keys(activePosition.head_deltas[selectedModel]).length === 0)) {
                      return;
                    }
                    setHighlightMode(newMode);
                  }}
                  style={{
                    padding: "2px 6px",
                    fontSize: 11,
                    borderRadius: 4,
                    border: "1px solid rgba(0,0,0,.2)",
                  }}
                >
                  <option value="attention">Attention</option>
                  <option value="value-weighted">Value-Weighted</option>
                  <option
                    value="delta"
                    disabled={!activePosition?.head_deltas || Object.keys(activePosition.head_deltas[selectedModel] || {}).length === 0}
                  >
                    Head Delta {(!activePosition?.head_deltas || Object.keys(activePosition.head_deltas[selectedModel] || {}).length === 0) ? "(enable in form)" : ""}
                  </option>
                </select>
              </label>
            </div>
          </div>

          {activePosition ? (
            <>
              <div style={{ fontSize: 14 }}>
                <strong>Context token:</strong> {activePosition.context_token.text || "␠"}{" "}
                → <strong>next:</strong> {activePosition.next_token.text || "␠"}
              </div>
              {activePosition.head_deltas && Object.keys(activePosition.head_deltas[selectedModel]).length > 0 && (
                <HeadInfluencePanel
                  position={activePosition}
                  selectedModel={selectedModel}
                  selectedLayer={selectedLayer}
                  selectedHead={selectedHead}
                  onHeadClick={(layer, head) => {
                    setSelectedLayer(layer);
                    setSelectedHead(head);
                  }}
                />
              )}
              <TopkPanel position={activePosition} />
              {/* <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                <AttnPanel 
                  label="Attention L1" 
                  attn={activePosition.attn.t1} 
                  valueWeightedAttn={activePosition.value_weighted_attn.t1}
                  showValueWeighted={showValueWeighted}
                />
                <AttnPanel 
                  label="Attention L2" 
                  attn={activePosition.attn.t2} 
                  valueWeightedAttn={activePosition.value_weighted_attn.t2}
                  showValueWeighted={showValueWeighted}
                />
              </div> */}
            </>
          ) : (
            <div style={{ opacity: 0.7 }}>hover a token to see logits & attention</div>
          )}

          <div style={{ opacity: 0.7, fontSize: 14 }}>
            device: {analysis.device}
          </div>
        </div>
      )}

      {!analysis && loading && (
        <div style={{ textAlign: "center", padding: 40, opacity: 0.7 }}>
          Loading analysis...
        </div>
      )}

    </div>
  );
}

const container = document.getElementById("root");
if (!container) throw new Error("missing #root");
const root = createRoot(container);
root.render(<App />);
