import { usePlaybackStore } from "../../stores/playbackStore";

const TEAM_COLORS = { A: "#e63946", B: "#457b9d" };

export default function TransitionsTab() {
  const { trackingData, setCurrentFrame } = usePlaybackStore();
  const transitions = trackingData?.transitions;

  if (!transitions || transitions.length === 0) {
    return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#484f58", fontSize: 12 }}>No transition data in JSON</div>;
  }

  const d2a_A = transitions.filter((t) => t.team === "A" && t.type === "defense_to_attack").length;
  const d2a_B = transitions.filter((t) => t.team === "B" && t.type === "defense_to_attack").length;
  const a2d_A = transitions.filter((t) => t.team === "A" && t.type === "attack_to_defense").length;
  const a2d_B = transitions.filter((t) => t.team === "B" && t.type === "attack_to_defense").length;

  return (
    <div style={{ display: "flex", gap: 32, height: "100%" }}>
      {/* Summary */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10, justifyContent: "center" }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: "#484f58", textTransform: "uppercase" }}>Counter attacks</div>
        <StatRow label="Def → Att" a={d2a_A} b={d2a_B} />
        <StatRow label="Att → Def" a={a2d_A} b={a2d_B} />
        <StatRow label="Total" a={d2a_A + a2d_A} b={d2a_B + a2d_B} />
      </div>

      {/* List */}
      <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: "#484f58", textTransform: "uppercase", marginBottom: 4 }}>Timeline</div>
        {transitions.map((t, i) => (
          <div
            key={i}
            onClick={() => setCurrentFrame(t.frame)}
            style={{ display: "flex", gap: 8, padding: "3px 0", fontSize: 11, cursor: "pointer", color: "#c9d1d9" }}
          >
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: TEAM_COLORS[t.team], flexShrink: 0, marginTop: 4 }} />
            <span style={{ width: 55, color: "#8b949e" }}>f{t.frame}</span>
            <span style={{ width: 80, fontSize: 10 }}>{t.type === "defense_to_attack" ? "Def→Att" : "Att→Def"}</span>
            <span style={{ color: t.outcome === "goal" ? "#4ade80" : "#8b949e" }}>{t.outcome ?? "—"}</span>
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
