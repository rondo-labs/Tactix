import { useRef, useEffect } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";

const TEAM_COLORS = { A: "#e63946", B: "#457b9d" };

export default function PassSonarTab() {
  const { trackingData, selectedPlayerIds } = usePlaybackStore();
  const pass_sonar = trackingData?.pass_sonar;

  if (!pass_sonar || Object.keys(pass_sonar).length === 0) {
    return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#484f58", fontSize: 12 }}>No pass sonar data in JSON</div>;
  }

  // Show selected players' sonars, or top 6 by total passes
  const entries = Object.entries(pass_sonar);
  let displayEntries: [string, typeof pass_sonar[string]][];

  if (selectedPlayerIds.length > 0) {
    displayEntries = entries.filter(([id]) => selectedPlayerIds.includes(Number(id)));
  } else {
    displayEntries = entries
      .sort(([, a], [, b]) => b.sectors.reduce((s, v) => s + v, 0) - a.sectors.reduce((s, v) => s + v, 0))
      .slice(0, 6);
  }

  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap", overflow: "auto", height: "100%", alignContent: "flex-start" }}>
      {displayEntries.map(([id, data]) => (
        <SonarChart key={id} playerId={id} sectors={data.sectors} team={data.team} />
      ))}
    </div>
  );
}

function SonarChart({ playerId, sectors, team }: { playerId: string; sectors: number[]; team: "A" | "B" }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const color = TEAM_COLORS[team] ?? "#888";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(cx, cy) - 8;

    ctx.clearRect(0, 0, w, h);

    const maxVal = Math.max(...sectors, 1);
    const sectorCount = sectors.length;
    const angleStep = (Math.PI * 2) / sectorCount;

    // Draw sectors
    for (let i = 0; i < sectorCount; i++) {
      const angle = -Math.PI / 2 + i * angleStep;
      const r = (sectors[i] / maxVal) * maxR;
      const x1 = cx + Math.cos(angle - angleStep / 2) * r;
      const y1 = cy + Math.sin(angle - angleStep / 2) * r;
      const x2 = cx + Math.cos(angle + angleStep / 2) * r;
      const y2 = cy + Math.sin(angle + angleStep / 2) * r;

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.closePath();
      ctx.fillStyle = color + "40";
      ctx.fill();
      ctx.strokeStyle = color + "80";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Guide circles
    for (const frac of [0.33, 0.66, 1]) {
      ctx.beginPath();
      ctx.arc(cx, cy, maxR * frac, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }, [sectors, color]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <canvas ref={canvasRef} width={100} height={100} />
      <span style={{ fontSize: 10, color: "#8b949e" }}>#{playerId} <span style={{ color }}>{sectors.reduce((s, v) => s + v, 0)} passes</span></span>
    </div>
  );
}
