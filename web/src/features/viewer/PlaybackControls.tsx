import { usePlaybackStore } from "../../stores/playbackStore";
import Timeline from "./Timeline";

const SPEEDS = [0.25, 0.5, 1, 2, 4];

const bar: React.CSSProperties = {
  display: "flex", flexDirection: "column", gap: 8,
  background: "#161b22", borderTop: "1px solid #30363d", padding: "10px 16px",
  flexShrink: 0,
};
const btn: React.CSSProperties = {
  padding: "4px 10px", background: "#0d1117", border: "1px solid #30363d",
  borderRadius: 4, fontSize: 13, color: "#e6edf3", cursor: "pointer",
};

export default function PlaybackControls() {
  const {
    currentFrame, isPlaying, playbackSpeed, trackingData,
    showVelocity, showIds,
    setCurrentFrame, togglePlaying, setPlaybackSpeed,
    toggleVelocity, toggleIds,
  } = usePlaybackStore();

  const totalFrames = trackingData?.meta.total_frames ?? 0;
  const fps = trackingData?.meta.fps ?? 25;

  const formatTime = (frame: number) => {
    const secs = frame / fps;
    const m = Math.floor(secs / 60);
    const s = (secs % 60).toFixed(1);
    return `${m}:${s.padStart(4, "0")}`;
  };

  return (
    <div style={bar}>
      {/* Timeline with event markers */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 11, color: "#8b949e", width: 180, flexShrink: 0 }}>
          {formatTime(currentFrame)} / {formatTime(totalFrames)} · frame {currentFrame}
        </span>
        <div style={{ flex: 1 }}>
          <Timeline />
        </div>
      </div>

      {/* Controls row */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <button style={btn} onClick={() => setCurrentFrame(Math.max(0, currentFrame - 1))}>‹</button>
        <button style={{ ...btn, background: "#58a6ff", color: "#000", fontWeight: 600, border: "none" }} onClick={togglePlaying}>
          {isPlaying ? "⏸" : "▶"}
        </button>
        <button style={btn} onClick={() => setCurrentFrame(Math.min(totalFrames - 1, currentFrame + 1))}>›</button>

        <select value={playbackSpeed} onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
          style={{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 4, padding: "3px 8px", fontSize: 11, color: "#e6edf3" }}>
          {SPEEDS.map((s) => <option key={s} value={s}>{s}x</option>)}
        </select>

        <span style={{ fontSize: 11, color: "#8b949e", marginLeft: 12 }}>Show:</span>
        <ToggleBtn label="Velocity" active={showVelocity} onClick={toggleVelocity} />
        <ToggleBtn label="IDs" active={showIds} onClick={toggleIds} />

        <span style={{ flex: 1 }} />
        <LegendDot color="#e63946" label="Team A" />
        <LegendDot color="#457b9d" label="Team B" />
        <LegendDot color="#a8dadc" label="GK" />
        <LegendDot color="#ffd60a" label="Ref" />
        <LegendDot color="#ffa500" label="Ball" />
      </div>
    </div>
  );
}

function ToggleBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} style={{
      padding: "2px 8px", borderRadius: 4, fontSize: 11, cursor: "pointer",
      background: active ? "rgba(88,166,255,0.15)" : "#0d1117",
      border: `1px solid ${active ? "rgba(88,166,255,0.4)" : "#30363d"}`,
      color: active ? "#58a6ff" : "#8b949e",
    }}>
      {label}
    </button>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "#8b949e" }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, display: "inline-block" }} />
      {label}
    </span>
  );
}
