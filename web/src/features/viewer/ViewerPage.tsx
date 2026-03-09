import { useRef, useCallback, useState, useEffect } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";
import Sidebar from "../sidebar/Sidebar";
import StatsPanel from "../stats/StatsPanel";
import PitchMinimap from "./PitchMinimap";
import PlaybackControls from "./PlaybackControls";
import { usePlaybackLoop } from "./usePlaybackLoop";

export default function ViewerPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const minimapWrapRef = useRef<HTMLDivElement>(null);
  const [minimapSize, setMinimapSize] = useState({ width: 600, height: 389 });
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [dividerPos, setDividerPos] = useState(50); // percentage for video pane

  const { trackingData, currentFrame, setTrackingData, isPlaying } =
    usePlaybackStore();

  usePlaybackLoop(videoRef);

  // Resize minimap to fit its container
  useEffect(() => {
    const el = minimapWrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      const aspect = 1559 / 1010;
      let w = width;
      let h = width / aspect;
      if (h > height) { h = height; w = height * aspect; }
      setMinimapSize({ width: Math.floor(w), height: Math.floor(h) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Sync video play/pause
  useEffect(() => {
    if (!videoRef.current) return;
    if (isPlaying) { videoRef.current.play().catch(() => {}); }
    else { videoRef.current.pause(); }
  }, [isPlaying]);

  const handleLoadJson = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => setTrackingData(JSON.parse(reader.result as string));
      reader.readAsText(file);
    },
    [setTrackingData]
  );

  const handleLoadVideo = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVideoUrl(URL.createObjectURL(file));
  }, []);

  // Divider drag
  const handleDividerDrag = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const container = (e.target as HTMLElement).parentElement!;
    const rect = container.getBoundingClientRect();
    const onMove = (ev: MouseEvent) => {
      const pct = ((ev.clientX - rect.left) / rect.width) * 100;
      setDividerPos(Math.min(80, Math.max(20, pct)));
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, []);

  const frame = trackingData?.frames[currentFrame] ?? null;

  return (
    <div style={{ display: "flex", height: "100vh", background: "#0d1117", color: "#e6edf3" }}>
      {/* Sidebar */}
      <Sidebar />

      {/* Main area */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Top bar */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 16px", background: "#161b22", borderBottom: "1px solid #30363d", flexShrink: 0 }}>
          <h1 style={{ fontSize: 14, fontWeight: 600, letterSpacing: "0.05em", color: "#58a6ff", textTransform: "uppercase" as const }}>
            Tactix
          </h1>
          <span style={{ flex: 1 }} />
          {trackingData && (
            <>
              <span style={{ fontSize: 11, color: "#8b949e", background: "#0d1117", border: "1px solid #30363d", borderRadius: 4, padding: "2px 8px" }}>
                {trackingData.meta.fps} fps
              </span>
              <span style={{ fontSize: 11, color: "#8b949e", background: "#0d1117", border: "1px solid #30363d", borderRadius: 4, padding: "2px 8px" }}>
                {trackingData.meta.total_frames} frames
              </span>
            </>
          )}
          <label style={{ padding: "5px 14px", background: "#0d1117", border: "1px solid #30363d", borderRadius: 6, fontSize: 12, cursor: "pointer", color: "#e6edf3" }}>
            Load Video
            <input type="file" accept="video/*" onChange={handleLoadVideo} style={{ display: "none" }} />
          </label>
          <label style={{ padding: "5px 14px", background: "#58a6ff", border: "none", borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: "pointer", color: "#000" }}>
            Load JSON
            <input type="file" accept=".json" onChange={handleLoadJson} style={{ display: "none" }} />
          </label>
        </div>

        {/* Dual pane: video + minimap with draggable divider */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative" }}>
          {/* Video pane */}
          <div style={{ width: `${dividerPos}%`, display: "flex", alignItems: "center", justifyContent: "center", background: "#000", padding: 12, overflow: "hidden" }}>
            {videoUrl ? (
              <video ref={videoRef} src={videoUrl} style={{ maxWidth: "100%", maxHeight: "100%", borderRadius: 6 }} muted playsInline />
            ) : (
              <div style={{ color: "#8b949e", fontSize: 14 }}>Load a video file to view</div>
            )}
          </div>

          {/* Draggable divider */}
          <div
            onMouseDown={handleDividerDrag}
            style={{
              width: 6, cursor: "col-resize", background: "#30363d", flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "background 0.15s",
            }}
            onMouseEnter={(e) => { (e.target as HTMLElement).style.background = "#58a6ff"; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.background = "#30363d"; }}
          >
            <div style={{ width: 2, height: 24, background: "#8b949e", borderRadius: 1 }} />
          </div>

          {/* Minimap pane */}
          <div ref={minimapWrapRef} style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 12, overflow: "hidden" }}>
            <PitchMinimap width={minimapSize.width} height={minimapSize.height} frame={frame} />
          </div>
        </div>

        {/* Playback controls */}
        <PlaybackControls />

        {/* Stats panel */}
        <StatsPanel />
      </div>
    </div>
  );
}
