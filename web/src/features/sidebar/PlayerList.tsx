import { usePlaybackStore } from "../../stores/playbackStore";
import type { TrackingPlayer } from "../../types/tracking";

const TEAM_COLORS: Record<string, string> = {
  A: "#e63946",
  B: "#457b9d",
  REFEREE: "#ffd60a",
  GOALKEEPER: "#a8dadc",
  UNKNOWN: "#888888",
};

export default function PlayerList() {
  const { trackingData, currentFrame, selectedPlayerIds, selectPlayer } = usePlaybackStore();

  const frame = trackingData?.frames[currentFrame];
  if (!frame) return <div style={{ fontSize: 12, color: "#8b949e", padding: 8 }}>No tracking data loaded</div>;

  const teamA: TrackingPlayer[] = [];
  const teamB: TrackingPlayer[] = [];
  const other: TrackingPlayer[] = [];

  for (const p of frame.players) {
    if (p.team === "A") teamA.push(p);
    else if (p.team === "B") teamB.push(p);
    else other.push(p);
  }

  const renderTeam = (label: string, color: string, players: TrackingPlayer[]) => {
    if (players.length === 0) return null;
    return (
      <div key={label}>
        <div style={{ fontSize: 10, fontWeight: 600, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4, display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, display: "inline-block" }} />
          {label}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
          {players.map((p) => {
            const selected = selectedPlayerIds.includes(p.id);
            return (
              <button
                key={p.id}
                onClick={(e) => selectPlayer(p.id, e.metaKey || e.ctrlKey)}
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "4px 8px", borderRadius: 4, border: "none",
                  background: selected ? "rgba(88,166,255,0.15)" : "transparent",
                  color: selected ? "#e6edf3" : "#c9d1d9",
                  fontSize: 12, cursor: "pointer", textAlign: "left",
                  width: "100%",
                }}
              >
                <span style={{ fontWeight: 600, minWidth: 28 }}>#{p.jersey ?? p.id}</span>
                <span style={{ color: "#8b949e", fontSize: 11, flex: 1 }}>
                  {p.speed != null ? `${p.speed.toFixed(1)} km/h` : "—"}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {renderTeam("Team A", TEAM_COLORS.A, teamA)}
      {renderTeam("Team B", TEAM_COLORS.B, teamB)}
      {renderTeam("Other", TEAM_COLORS.UNKNOWN, other)}
    </div>
  );
}
