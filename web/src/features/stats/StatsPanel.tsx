import { useState } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";

const TABS = ["Formation", "Shots", "Passes", "Zone 14", "Buildup", "Duels", "Transitions", "Set Pieces"] as const;

export default function StatsPanel() {
  const { bottomPanelOpen, toggleBottomPanel } = usePlaybackStore();
  const [activeTab, setActiveTab] = useState<string>(TABS[0]);

  return (
    <div style={{ background: "#161b22", borderTop: "1px solid #30363d", flexShrink: 0 }}>
      {/* Toggle bar — always visible */}
      <div
        onClick={toggleBottomPanel}
        style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "6px 16px", cursor: "pointer", userSelect: "none",
        }}
      >
        <span style={{ fontSize: 11, fontWeight: 600, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          Statistics
        </span>
        <span style={{ fontSize: 10, color: "#8b949e", transform: bottomPanelOpen ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}>
          ▲
        </span>
        {bottomPanelOpen && (
          <div style={{ display: "flex", gap: 2, marginLeft: 12 }} onClick={(e) => e.stopPropagation()}>
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setActiveTab(t)}
                style={{
                  padding: "3px 10px", fontSize: 11, borderRadius: 4, cursor: "pointer",
                  border: `1px solid ${activeTab === t ? "rgba(88,166,255,0.4)" : "#30363d"}`,
                  background: activeTab === t ? "rgba(88,166,255,0.12)" : "transparent",
                  color: activeTab === t ? "#58a6ff" : "#8b949e",
                }}
              >
                {t}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Panel content */}
      {bottomPanelOpen && (
        <div style={{ height: 240, padding: "12px 16px", overflow: "auto", borderTop: "1px solid #30363d" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#8b949e", fontSize: 13 }}>
            {activeTab} — coming soon
          </div>
        </div>
      )}
    </div>
  );
}
