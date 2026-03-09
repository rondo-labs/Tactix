import { usePlaybackStore, LAYER_GROUPS } from "../../stores/playbackStore";

export default function LayerToggles() {
  const { layers, toggleLayer } = usePlaybackStore();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {LAYER_GROUPS.map((group) => (
        <div key={group.label}>
          <div style={{ fontSize: 10, fontWeight: 600, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
            {group.label}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {group.layers.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => toggleLayer(key)}
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "5px 8px", borderRadius: 4, border: "none",
                  background: layers[key] ? "rgba(88,166,255,0.12)" : "transparent",
                  color: layers[key] ? "#58a6ff" : "#c9d1d9",
                  fontSize: 12, cursor: "pointer", textAlign: "left",
                  transition: "background 0.15s",
                }}
              >
                <span style={{
                  width: 14, height: 14, borderRadius: 3, flexShrink: 0,
                  border: `1.5px solid ${layers[key] ? "#58a6ff" : "#30363d"}`,
                  background: layers[key] ? "#58a6ff" : "transparent",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 10, color: "#000", fontWeight: 700,
                }}>
                  {layers[key] ? "✓" : ""}
                </span>
                {label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
