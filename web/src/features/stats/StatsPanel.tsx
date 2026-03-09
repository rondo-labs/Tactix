import { useState } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";
import FormationTab from "./FormationTab";
import ShotMapTab from "./ShotMapTab";
import PassSonarTab from "./PassSonarTab";
import Zone14Tab from "./Zone14Tab";
import BuildupTab from "./BuildupTab";
import DuelsTab from "./DuelsTab";
import TransitionsTab from "./TransitionsTab";
import SetPiecesTab from "./SetPiecesTab";

const TABS = [
  { key: "formation", label: "Formation", Component: FormationTab },
  { key: "shots", label: "Shots", Component: ShotMapTab },
  { key: "passes", label: "Passes", Component: PassSonarTab },
  { key: "zone14", label: "Zone 14", Component: Zone14Tab },
  { key: "buildup", label: "Buildup", Component: BuildupTab },
  { key: "duels", label: "Duels", Component: DuelsTab },
  { key: "transitions", label: "Transitions", Component: TransitionsTab },
  { key: "setPieces", label: "Set Pieces", Component: SetPiecesTab },
] as const;

export default function StatsPanel() {
  const { bottomPanelOpen, toggleBottomPanel } = usePlaybackStore();
  const [activeTab, setActiveTab] = useState<string>(TABS[0].key);

  const ActiveComponent = TABS.find((t) => t.key === activeTab)?.Component ?? FormationTab;

  return (
    <div style={{ background: "#161b22", borderTop: "1px solid #30363d", flexShrink: 0 }}>
      {/* Toggle bar */}
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
                key={t.key}
                onClick={() => setActiveTab(t.key)}
                style={{
                  padding: "3px 10px", fontSize: 11, borderRadius: 4, cursor: "pointer",
                  border: `1px solid ${activeTab === t.key ? "rgba(88,166,255,0.4)" : "#30363d"}`,
                  background: activeTab === t.key ? "rgba(88,166,255,0.12)" : "transparent",
                  color: activeTab === t.key ? "#58a6ff" : "#8b949e",
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Panel content */}
      {bottomPanelOpen && (
        <div style={{ height: 240, padding: "12px 16px", overflow: "auto", borderTop: "1px solid #30363d" }}>
          <ActiveComponent />
        </div>
      )}
    </div>
  );
}
