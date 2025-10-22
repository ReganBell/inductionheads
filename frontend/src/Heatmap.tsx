import React, { useEffect, useRef } from "react";

type HeatmapProps = {
  matrix: number[][];
  size?: number;
  title?: string;
};

export function Heatmap({ matrix, size = 160, title }: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rows = matrix.length;
    const cols = matrix[0]?.length ?? 0;
    if (rows === 0 || cols === 0) return;

    canvas.width = size;
    canvas.height = size;
    const cellW = size / cols;
    const cellH = size / rows;

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const v = Math.max(0, Math.min(1, matrix[r][c] ?? 0));
        const shade = Math.floor(v * 255);
        ctx.fillStyle = `rgb(${shade}, ${shade}, 255)`;
        ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    ctx.strokeStyle = "rgba(0,0,0,.15)";
    ctx.strokeRect(0, 0, size, size);
  }, [matrix, size]);

  return (
    <div style={{ display: "inline-flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size, imageRendering: "pixelated", borderRadius: 6 }}
      />
      {title && <div style={{ fontSize: 12, opacity: 0.7 }}>{title}</div>}
    </div>
  );
}
