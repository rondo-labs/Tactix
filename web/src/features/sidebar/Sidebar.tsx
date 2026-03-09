import { useState } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";
import LayerToggles from "./LayerToggles";
import PlayerList from "./PlayerList";
import PlayerDetail from "./PlayerDetail";

type Tab = "layers" | "players";

export default function Sidebar() {
  const { sidebarOpen, toggleSidebar, selectedPlayerIds } = usePlaybackStore();
  const [tab, setTab] = useState<Tab>("layers");

  if (!sidebarOpen) {
    return (
      <div style={{ width: 48, background: "#161b22", borderRight: "1px solid #30363d", display: "flex", flexDirection: "column", alignItems: "center", paddingTop: 8, flexShrink: 0 }}>
        <button onClick={toggleSidebar} style={{ background: "none", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 18, padding: 8 }} title="Open sidebar (S)">
          ☰
        </button>
      </div>
    );
  }

  const showDetail = selectedPlayerIds.length === 1;

  return (
    <div style={{ width: 260, background: "#161b22", borderRight: "1px solid #30363d", display: "flex", flexDirection: "column", flexShrink: 0 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", padding: "8px 12px", borderBottom: "1px solid #30363d" }}>
        <div style={{ display: "flex", gap: 0, flex: 1 }}>
          {(["layers", "players"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: "4px 12px", fontSize: 11, fontWeight: 600, cursor: "pointer",
                border: "1px solid #30363d", borderRadius: t === "layers" ? "4px 0 0 4px" : "0 4px 4px 0",
                background: tab === t ? "#0d1117" : "transparent",
                color: tab === t ? "#e6edf3" : "#8b949e",
              }}
            >
              {t === "layers" ? "Layers" : "Players"}
            </button>
          ))}
        </div>
        <button onClick={toggleSidebar} style={{ background: "none", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 14, padding: "2px 4px" }} title="Close sidebar (S)">
          ✕
        </button>
      </div>

      {/* Player detail panel (when single player selected) */}
      {showDetail && (
        <div style={{ padding: 12, borderBottom: "1px solid #30363d" }}>
          <PlayerDetail />
        </div>
      )}

      {/* Content */}
      <div style={{ flex: 1, overflow: "auto", padding: 12 }}>
        {tab === "layers" ? <LayerToggles /> : <PlayerList />}
      </div>

      {/* Shortcut hints */}
      <div style={{ padding: "6px 12px", borderTop: "1px solid #21262d", fontSize: 9, color: "#484f58", display: "flex", flexWrap: "wrap", gap: 6 }}>
        <span>Space: play</span>
        <span>←→: frame</span>
        <span>↑↓: speed</span>
        <span>V: velocity</span>
        <span>I: IDs</span>
        <span>B: stats</span>
        <span>Esc: deselect</span>
      </div>
    </div>
  );
}
