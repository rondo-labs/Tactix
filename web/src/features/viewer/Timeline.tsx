import { useRef, useCallback, useState } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";
import type { TrackingEvent } from "../../types/tracking";

const EVENT_COLORS: Record<string, string> = {
  goal: "#4ade80",
  shot: "#f97316",
  corner: "#a78bfa",
  free_kick: "#60a5fa",
  foul: "#fbbf24",
  offside: "#f87171",
  card: "#ef4444",
  substitution: "#8b949e",
  other: "#8b949e",
};

export default function Timeline() {
  const { trackingData, currentFrame, setCurrentFrame } = usePlaybackStore();
  const trackRef = useRef<HTMLDivElement>(null);
  const [hoveredEvent, setHoveredEvent] = useState<TrackingEvent | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const totalFrames = trackingData?.meta.total_frames ?? 1;
  const events = trackingData?.events ?? [];

  const handleTrackClick = useCallback(
    (e: React.MouseEvent) => {
      const rect = trackRef.current?.getBoundingClientRect();
      if (!rect) return;
      const pct = (e.clientX - rect.left) / rect.width;
      setCurrentFrame(Math.round(pct * (totalFrames - 1)));
    },
    [totalFrames, setCurrentFrame]
  );

  const handleTrackDrag = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const rect = trackRef.current?.getBoundingClientRect();
      if (!rect) return;
      const update = (ev: MouseEvent) => {
        const pct = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
        setCurrentFrame(Math.round(pct * (totalFrames - 1)));
      };
      const up = () => {
        window.removeEventListener("mousemove", update);
        window.removeEventListener("mouseup", up);
      };
      update(e.nativeEvent);
      window.addEventListener("mousemove", update);
      window.addEventListener("mouseup", up);
    },
    [totalFrames, setCurrentFrame]
  );

  const progressPct = totalFrames > 1 ? (currentFrame / (totalFrames - 1)) * 100 : 0;

  return (
    <div style={{ position: "relative" }}>
      {/* Track */}
      <div
        ref={trackRef}
        onMouseDown={handleTrackDrag}
        onClick={handleTrackClick}
        style={{
          position: "relative", height: 28, background: "#0d1117",
          borderRadius: 4, cursor: "pointer", overflow: "hidden",
          border: "1px solid #30363d",
        }}
      >
        {/* Progress fill */}
        <div style={{
          position: "absolute", top: 0, left: 0, bottom: 0,
          width: `${progressPct}%`, background: "rgba(88,166,255,0.15)",
        }} />

        {/* Playhead */}
        <div style={{
          position: "absolute", top: 0, bottom: 0, left: `${progressPct}%`,
          width: 2, background: "#58a6ff", transform: "translateX(-1px)",
        }} />

        {/* Event markers */}
        {events.map((ev, i) => {
          const pct = (ev.frame / (totalFrames - 1)) * 100;
          const color = EVENT_COLORS[ev.type] ?? EVENT_COLORS.other;
          return (
            <div
              key={i}
              onMouseEnter={(e) => { setHoveredEvent(ev); setTooltipPos({ x: e.clientX, y: e.clientY }); }}
              onMouseLeave={() => setHoveredEvent(null)}
              onClick={(e) => { e.stopPropagation(); setCurrentFrame(ev.frame); }}
              style={{
                position: "absolute", top: 2, bottom: 2, left: `${pct}%`,
                width: 6, transform: "translateX(-3px)", borderRadius: 2,
                background: color, opacity: 0.85, cursor: "pointer",
                zIndex: 2,
              }}
            />
          );
        })}
      </div>

      {/* Tooltip */}
      {hoveredEvent && (
        <div style={{
          position: "fixed", left: tooltipPos.x + 8, top: tooltipPos.y - 40,
          background: "#1c2128", border: "1px solid #30363d", borderRadius: 6,
          padding: "6px 10px", fontSize: 11, color: "#e6edf3", zIndex: 1000,
          pointerEvents: "none", whiteSpace: "nowrap",
        }}>
          <span style={{ color: EVENT_COLORS[hoveredEvent.type] ?? "#8b949e", fontWeight: 600 }}>
            {hoveredEvent.type}
          </span>
          {hoveredEvent.team && <span style={{ color: "#8b949e" }}> · Team {hoveredEvent.team}</span>}
          {hoveredEvent.outcome && <span style={{ color: "#8b949e" }}> · {hoveredEvent.outcome}</span>}
          {hoveredEvent.label && <span style={{ color: "#8b949e" }}> · {hoveredEvent.label}</span>}
          <span style={{ color: "#484f58", marginLeft: 8 }}>frame {hoveredEvent.frame}</span>
        </div>
      )}
    </div>
  );
}
