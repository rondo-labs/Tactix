import { usePlaybackStore } from "../../stores/playbackStore";

const TEAM_COLORS = { A: "#e63946", B: "#457b9d" };

export default function SetPiecesTab() {
  const { trackingData, setCurrentFrame } = usePlaybackStore();
  const sp = trackingData?.set_pieces;

  if (!sp) {
    return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#484f58", fontSize: 12 }}>No set piece data in JSON</div>;
  }

  const corners = sp.corners ?? [];
  const freeKicks = sp.free_kicks ?? [];
  const cornersA = corners.filter((c) => c.team === "A").length;
  const cornersB = corners.filter((c) => c.team === "B").length;
  const fkA = freeKicks.filter((f) => f.team === "A").length;
  const fkB = freeKicks.filter((f) => f.team === "B").length;

  return (
    <div style={{ display: "flex", gap: 32, height: "100%" }}>
      {/* Summary */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10, justifyContent: "center" }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: "#484f58", textTransform: "uppercase" }}>Set pieces</div>
        <StatRow label="Corners" a={cornersA} b={cornersB} />
        <StatRow label="Free kicks" a={fkA} b={fkB} />
        <StatRow label="Total" a={cornersA + fkA} b={cornersB + fkB} />
      </div>

      {/* Event list */}
      <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: "#484f58", textTransform: "uppercase", marginBottom: 4 }}>Events</div>
        {[...corners.map((c) => ({ ...c, kind: "Corner" as const })), ...freeKicks.map((f) => ({ ...f, kind: "Free kick" as const }))]
          .sort((a, b) => a.frame - b.frame)
          .map((ev, i) => (
            <div
              key={i}
              onClick={() => setCurrentFrame(ev.frame)}
              style={{ display: "flex", gap: 8, padding: "3px 0", fontSize: 11, cursor: "pointer", color: "#c9d1d9" }}
            >
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: TEAM_COLORS[ev.team], flexShrink: 0, marginTop: 4 }} />
              <span style={{ width: 55, color: "#8b949e" }}>f{ev.frame}</span>
              <span style={{ width: 70 }}>{ev.kind}</span>
              <span style={{ color: "#8b949e" }}>{ev.outcome ?? "—"}</span>
            </div>
          ))}
      </div>
    </div>
  );
}

function StatRow({ label, a, b }: { label: string; a: number; b: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12 }}>
      <span style={{ width: 24, textAlign: "right", color: TEAM_COLORS.A, fontWeight: 600 }}>{a}</span>
      <span style={{ width: 70, textAlign: "center", color: "#8b949e" }}>{label}</span>
      <span style={{ width: 24, textAlign: "left", color: TEAM_COLORS.B, fontWeight: 600 }}>{b}</span>
    </div>
  );
}
