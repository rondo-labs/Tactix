import { useRef, useEffect } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";

const TEAM_COLORS = { A: "#e63946", B: "#457b9d" };

export default function Zone14Tab() {
  const { trackingData } = usePlaybackStore();
  const zone14 = trackingData?.zone14;

  if (!zone14) {
    return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#484f58", fontSize: 12 }}>No Zone 14 data in JSON</div>;
  }

  return (
    <div style={{ display: "flex", gap: 24, height: "100%", alignItems: "center", justifyContent: "center" }}>
      {(["A", "B"] as const).map((team) => {
        const grid = zone14[team];
        if (!grid) return null;
        return <ZoneGrid key={team} team={team} grid={grid} color={TEAM_COLORS[team]} />;
      })}
    </div>
  );
}

function ZoneGrid({ team, grid, color }: { team: string; grid: number[][]; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const rows = grid.length;
    const cols = grid[0]?.length ?? 0;
    if (rows === 0 || cols === 0) return;

    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    let maxVal = 0;
    for (const row of grid) for (const v of row) if (v > maxVal) maxVal = v;
    if (maxVal === 0) maxVal = 1;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = grid[r][c] / maxVal;
        ctx.fillStyle = color;
        ctx.globalAlpha = val * 0.7 + 0.03;
        ctx.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);
      }
    }
    ctx.globalAlpha = 1;

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 0.5;
    for (let c = 0; c <= cols; c++) {
      ctx.beginPath(); ctx.moveTo(c * cellW, 0); ctx.lineTo(c * cellW, canvas.height); ctx.stroke();
    }
    for (let r = 0; r <= rows; r++) {
      ctx.beginPath(); ctx.moveTo(0, r * cellH); ctx.lineTo(canvas.width, r * cellH); ctx.stroke();
    }
  }, [grid, color]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <span style={{ fontSize: 11, color: "#8b949e" }}>Team {team} — Zone 14 penetration</span>
      <canvas ref={canvasRef} width={200} height={200} style={{ borderRadius: 4, border: "1px solid #30363d" }} />
    </div>
  );
}
