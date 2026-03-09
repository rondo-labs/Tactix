import { usePlaybackStore } from "../../stores/playbackStore";

const TEAM_COLORS = { A: "#e63946", B: "#457b9d" };

export default function BuildupTab() {
  const { trackingData, setCurrentFrame } = usePlaybackStore();
  const buildups = trackingData?.buildups;

  if (!buildups || buildups.length === 0) {
    return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#484f58", fontSize: 12 }}>No buildup data in JSON</div>;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, overflow: "auto", height: "100%" }}>
      <div style={{ display: "flex", gap: 8, padding: "4px 0", fontSize: 10, fontWeight: 600, color: "#484f58", borderBottom: "1px solid #21262d" }}>
        <span style={{ width: 40 }}>#</span>
        <span style={{ width: 50 }}>Team</span>
        <span style={{ width: 60 }}>Passes</span>
        <span style={{ width: 80 }}>Frames</span>
        <span style={{ flex: 1 }}>Outcome</span>
      </div>
      {buildups.map((b) => (
        <div
          key={b.id}
          onClick={() => setCurrentFrame(b.frame_start)}
          style={{
            display: "flex", gap: 8, padding: "4px 0", fontSize: 11, cursor: "pointer",
            color: "#c9d1d9", borderBottom: "1px solid #21262d",
          }}
        >
          <span style={{ width: 40, color: "#8b949e" }}>{b.id}</span>
          <span style={{ width: 50 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: TEAM_COLORS[b.team], display: "inline-block", marginRight: 4 }} />
            {b.team}
          </span>
          <span style={{ width: 60 }}>{b.passes.length}</span>
          <span style={{ width: 80, color: "#8b949e" }}>{b.frame_start}–{b.frame_end}</span>
          <span style={{ flex: 1, color: b.outcome === "goal" ? "#4ade80" : "#8b949e" }}>{b.outcome ?? "—"}</span>
        </div>
      ))}
    </div>
  );
}
