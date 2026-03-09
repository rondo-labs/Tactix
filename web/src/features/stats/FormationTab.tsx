import { useRef, useEffect } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";

const TEAM_COLORS = { A: "#e63946", B: "#457b9d" };

export default function FormationTab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { trackingData, currentFrame } = usePlaybackStore();

  const formations = trackingData?.formations;

  // Find active formations for current frame
  const activeA = formations?.find((f) => f.team === "A" && currentFrame >= f.frame_start && currentFrame <= f.frame_end);
  const activeB = formations?.find((f) => f.team === "B" && currentFrame >= f.frame_start && currentFrame <= f.frame_end);

  // Draw formation diagram
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const drawFormation = (lines: number[], offsetX: number, halfW: number, color: string) => {
      const totalLines = lines.length + 1; // +1 for GK
      const allLines = [1, ...lines]; // GK + outfield lines
      for (let row = 0; row < allLines.length; row++) {
        const count = allLines[row];
        const y = 20 + ((row) / (totalLines)) * (h - 40);
        for (let col = 0; col < count; col++) {
          const x = offsetX + (halfW / (count + 1)) * (col + 1);
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = "rgba(255,255,255,0.5)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    };

    if (activeA?.lines) drawFormation(activeA.lines, 0, w / 2 - 10, TEAM_COLORS.A);
    if (activeB?.lines) drawFormation(activeB.lines, w / 2 + 10, w / 2 - 10, TEAM_COLORS.B);
  }, [activeA, activeB]);

  if (!formations || formations.length === 0) {
    return <Empty text="No formation data in JSON" />;
  }

  return (
    <div style={{ display: "flex", gap: 24, height: "100%" }}>
      {/* Formation diagram */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
          <FormationBadge label={activeA?.name ?? "—"} color={TEAM_COLORS.A} team="A" />
          <FormationBadge label={activeB?.name ?? "—"} color={TEAM_COLORS.B} team="B" />
        </div>
        <canvas ref={canvasRef} width={360} height={180} style={{ background: "rgba(255,255,255,0.03)", borderRadius: 6, flex: 1 }} />
      </div>

      {/* Formation timeline */}
      <div style={{ width: 240, overflow: "auto" }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: "#8b949e", textTransform: "uppercase", marginBottom: 6 }}>Changes</div>
        {formations.map((f, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0", fontSize: 11, color: "#c9d1d9" }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: TEAM_COLORS[f.team], flexShrink: 0 }} />
            <span style={{ color: "#8b949e" }}>f{f.frame_start}</span>
            <span style={{ fontWeight: 600 }}>{f.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function FormationBadge({ label, color, team }: { label: string; color: string; team: string }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
      <span style={{ color: "#8b949e" }}>Team {team}</span>
      <span style={{ fontWeight: 600, color: "#e6edf3" }}>{label}</span>
    </span>
  );
}

function Empty({ text }: { text: string }) {
  return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#484f58", fontSize: 12 }}>{text}</div>;
}
