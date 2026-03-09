import { useRef, useEffect } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";

const TEAM_COLORS = { A: "#e63946", B: "#457b9d" };
const PITCH = { L: 105, W: 68 };

export default function ShotMapTab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { trackingData } = usePlaybackStore();
  const shots = trackingData?.shots;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !shots) return;
    const ctx = canvas.getContext("2d")!;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const margin = 10;
    const pw = w - margin * 2;
    const ph = h - margin * 2;
    const mx = (m: number) => margin + (m / PITCH.L) * pw;
    const my = (m: number) => margin + (m / PITCH.W) * ph;

    // Half-pitch background
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    ctx.fillRect(margin, margin, pw, ph);

    // Pitch outline
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.strokeRect(margin, margin, pw, ph);

    // Penalty area (right side)
    ctx.strokeRect(mx(PITCH.L - 16.5), my((PITCH.W - 40.32) / 2), (16.5 / PITCH.L) * pw, (40.32 / PITCH.W) * ph);

    // Goal
    ctx.strokeRect(mx(PITCH.L), my((PITCH.W - 7.32) / 2), (2 / PITCH.L) * pw, (7.32 / PITCH.W) * ph);

    // Shots
    for (const s of shots) {
      const sx = mx(s.x);
      const sy = my(s.y);
      const color = TEAM_COLORS[s.team] ?? "#888";
      const r = s.xg != null ? 4 + s.xg * 16 : 6;

      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = s.outcome === "goal" ? color : "transparent";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = s.on_target ? 2 : 1;
      if (!s.on_target) ctx.setLineDash([2, 2]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [shots]);

  if (!shots || shots.length === 0) {
    return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#484f58", fontSize: 12 }}>No shot data in JSON</div>;
  }

  const goalsA = shots.filter((s) => s.team === "A" && s.outcome === "goal").length;
  const goalsB = shots.filter((s) => s.team === "B" && s.outcome === "goal").length;
  const shotsA = shots.filter((s) => s.team === "A").length;
  const shotsB = shots.filter((s) => s.team === "B").length;
  const onTargetA = shots.filter((s) => s.team === "A" && s.on_target).length;
  const onTargetB = shots.filter((s) => s.team === "B" && s.on_target).length;

  return (
    <div style={{ display: "flex", gap: 20, height: "100%" }}>
      <canvas ref={canvasRef} width={420} height={200} style={{ borderRadius: 6, flexShrink: 0 }} />
      <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 11, justifyContent: "center" }}>
        <StatRow label="Goals" a={goalsA} b={goalsB} />
        <StatRow label="Shots" a={shotsA} b={shotsB} />
        <StatRow label="On target" a={onTargetA} b={onTargetB} />
        <div style={{ marginTop: 8, color: "#484f58", fontSize: 10 }}>
          ● filled = goal &nbsp; ○ solid = on target &nbsp; ○ dashed = off target
          <br />Circle size = xG
        </div>
      </div>
    </div>
  );
}

function StatRow({ label, a, b }: { label: string; a: number; b: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ width: 24, textAlign: "right", color: TEAM_COLORS.A, fontWeight: 600 }}>{a}</span>
      <span style={{ width: 80, textAlign: "center", color: "#8b949e" }}>{label}</span>
      <span style={{ width: 24, textAlign: "left", color: TEAM_COLORS.B, fontWeight: 600 }}>{b}</span>
    </div>
  );
}
