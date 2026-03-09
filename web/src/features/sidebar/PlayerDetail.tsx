import { useMemo, useRef, useEffect } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";

const TEAM_COLORS: Record<string, string> = { A: "#e63946", B: "#457b9d", REFEREE: "#ffd60a", REF: "#ffd60a", GOALKEEPER: "#a8dadc", GK: "#a8dadc", UNKNOWN: "#888", "?": "#888" };

export default function PlayerDetail() {
  const { trackingData, currentFrame, selectedPlayerIds, clearSelectedPlayers } = usePlaybackStore();

  const playerId = selectedPlayerIds[0];

  const stats = useMemo(() => {
    if (!trackingData || playerId == null) return null;

    let team = "UNKNOWN";
    let jersey: string | undefined;
    let totalDist = 0;
    let maxSpeed = 0;
    let speedSum = 0;
    let speedCount = 0;
    const positions: { x: number; y: number }[] = [];
    let prevX: number | null = null;
    let prevY: number | null = null;

    const srcW = trackingData.meta.canvas.width;
    const srcH = trackingData.meta.canvas.height;

    const end = Math.min(currentFrame, trackingData.frames.length - 1);
    for (let f = 0; f <= end; f++) {
      const frame = trackingData.frames[f];
      const p = frame.players.find((pl) => pl.id === playerId);
      if (!p) continue;

      team = p.team;
      jersey = p.jersey;

      if (prevX != null && prevY != null) {
        const dx = (p.x - prevX) / srcW * 105;
        const dy = (p.y - prevY) / srcH * 68;
        totalDist += Math.sqrt(dx * dx + dy * dy);
      }
      prevX = p.x;
      prevY = p.y;

      if (p.speed != null) {
        if (p.speed > maxSpeed) maxSpeed = p.speed;
        speedSum += p.speed;
        speedCount++;
      }

      if (f % 5 === 0) positions.push({ x: p.x, y: p.y });
    }

    return { team, jersey, totalDist, maxSpeed, avgSpeed: speedCount > 0 ? speedSum / speedCount : 0, positions, srcW, srcH };
  }, [trackingData, currentFrame, playerId]);

  if (!stats) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: TEAM_COLORS[stats.team] ?? "#888" }} />
          <span style={{ fontSize: 14, fontWeight: 700, color: "#e6edf3" }}>#{stats.jersey ?? playerId}</span>
          <span style={{ fontSize: 11, color: "#8b949e" }}>Team {stats.team}</span>
        </div>
        <button onClick={clearSelectedPlayers} style={{ background: "none", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 12 }}>
          ✕
        </button>
      </div>

      {/* Stats grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
        <StatCard label="Distance" value={`${(stats.totalDist / 1000).toFixed(2)} km`} />
        <StatCard label="Max speed" value={`${stats.maxSpeed.toFixed(1)} km/h`} />
        <StatCard label="Avg speed" value={`${stats.avgSpeed.toFixed(1)} km/h`} />
      </div>

      {/* Mini trail map */}
      <TrailCanvas positions={stats.positions} team={stats.team} srcW={stats.srcW} srcH={stats.srcH} />
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ background: "#0d1117", border: "1px solid #21262d", borderRadius: 6, padding: "8px 10px" }}>
      <div style={{ fontSize: 9, color: "#484f58", textTransform: "uppercase", fontWeight: 600, marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 13, color: "#e6edf3", fontWeight: 600 }}>{value}</div>
    </div>
  );
}

function TrailCanvas({ positions, team, srcW, srcH }: { positions: { x: number; y: number }[]; team: string; srcW: number; srcH: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || positions.length < 2) return;
    const ctx = canvas.getContext("2d")!;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = "rgba(255,255,255,0.02)";
    ctx.fillRect(0, 0, w, h);

    const color = TEAM_COLORS[team] ?? "#888";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.6;
    ctx.beginPath();

    for (let i = 0; i < positions.length; i++) {
      const px = (positions[i].x / srcW) * w;
      const py = (positions[i].y / srcH) * h;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Current position dot
    const last = positions[positions.length - 1];
    ctx.beginPath();
    ctx.arc((last.x / srcW) * w, (last.y / srcH) * h, 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }, [positions, team, srcW, srcH]);

  return <canvas ref={canvasRef} width={220} height={143} style={{ borderRadius: 4, border: "1px solid #21262d" }} />;
}
