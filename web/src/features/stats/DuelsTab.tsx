import { useRef, useEffect } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";

const TEAM_COLORS = { A: "#e63946", B: "#457b9d" };

export default function DuelsTab() {
  const { trackingData } = usePlaybackStore();
  const duels = trackingData?.duels;

  if (!duels) {
    return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#484f58", fontSize: 12 }}>No duel data in JSON</div>;
  }

  return (
    <div style={{ display: "flex", gap: 24, height: "100%", alignItems: "center", justifyContent: "center" }}>
      <DuelGrid label="Team A wins" grid={duels.grid_a} color={TEAM_COLORS.A} />
      <DuelGrid label="Team B wins" grid={duels.grid_b} color={TEAM_COLORS.B} />
      <DuelStats gridA={duels.grid_a} gridB={duels.grid_b} />
    </div>
  );
}

function DuelGrid({ label, grid, color }: { label: string; grid: number[][]; color: string }) {
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
        ctx.globalAlpha = val * 0.8 + 0.02;
        ctx.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);
      }
    }
    ctx.globalAlpha = 1;
  }, [grid, color]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <span style={{ fontSize: 11, color: "#8b949e" }}>{label}</span>
      <canvas ref={canvasRef} width={180} height={120} style={{ borderRadius: 4, border: "1px solid #30363d" }} />
    </div>
  );
}

function DuelStats({ gridA, gridB }: { gridA: number[][]; gridB: number[][] }) {
  const sumGrid = (g: number[][]) => g.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0);
  const totalA = sumGrid(gridA);
  const totalB = sumGrid(gridB);
  const total = totalA + totalB || 1;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
      <div style={{ color: "#8b949e", fontSize: 10, fontWeight: 600, textTransform: "uppercase" }}>Win rate</div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: TEAM_COLORS.A, fontWeight: 600 }}>{totalA}</span>
        <div style={{ width: 120, height: 8, borderRadius: 4, background: "#21262d", overflow: "hidden", display: "flex" }}>
          <div style={{ width: `${(totalA / total) * 100}%`, background: TEAM_COLORS.A }} />
          <div style={{ width: `${(totalB / total) * 100}%`, background: TEAM_COLORS.B }} />
        </div>
        <span style={{ color: TEAM_COLORS.B, fontWeight: 600 }}>{totalB}</span>
      </div>
      <span style={{ fontSize: 10, color: "#484f58" }}>{Math.round((totalA / total) * 100)}% — {Math.round((totalB / total) * 100)}%</span>
    </div>
  );
}
