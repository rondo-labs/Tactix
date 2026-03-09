import { create } from "zustand";
import type { TrackingData } from "../types/tracking";

export type LayerKey =
  | "voronoi" | "heatmap" | "passNetwork" | "pressure" | "coverShadow"
  | "formation" | "teamCentroid" | "teamWidthDepth" | "compactness"
  | "shotMap" | "passSonar" | "zone14" | "buildupSequence"
  | "duelHeatmap" | "transitions" | "setPieces";

export const LAYER_GROUPS: { label: string; layers: { key: LayerKey; label: string }[] }[] = [
  {
    label: "Base Overlays",
    layers: [
      { key: "voronoi", label: "Voronoi Territory" },
      { key: "heatmap", label: "Heatmap" },
      { key: "passNetwork", label: "Pass Network" },
      { key: "pressure", label: "Pressure Index" },
      { key: "coverShadow", label: "Cover Shadow" },
    ],
  },
  {
    label: "Team Metrics",
    layers: [
      { key: "formation", label: "Formation" },
      { key: "teamCentroid", label: "Team Centroid" },
      { key: "teamWidthDepth", label: "Width / Depth" },
      { key: "compactness", label: "Compactness" },
    ],
  },
  {
    label: "Attack Analysis",
    layers: [
      { key: "shotMap", label: "Shot Map" },
      { key: "passSonar", label: "Pass Sonar" },
      { key: "zone14", label: "Zone 14" },
      { key: "buildupSequence", label: "Buildup Sequence" },
    ],
  },
  {
    label: "Defence / Transitions",
    layers: [
      { key: "duelHeatmap", label: "Duel Heatmap" },
      { key: "transitions", label: "Transitions" },
      { key: "setPieces", label: "Set Pieces" },
    ],
  },
];

interface PlaybackState {
  trackingData: TrackingData | null;
  currentFrame: number;
  isPlaying: boolean;
  playbackSpeed: number;
  showVelocity: boolean;
  showTrails: boolean;
  showIds: boolean;

  // Sidebar
  sidebarOpen: boolean;
  layers: Record<LayerKey, boolean>;
  selectedPlayerIds: number[];

  // Bottom panel
  bottomPanelOpen: boolean;
  bottomPanelHeight: number;

  setTrackingData: (data: TrackingData) => void;
  setCurrentFrame: (frame: number) => void;
  setIsPlaying: (playing: boolean) => void;
  togglePlaying: () => void;
  setPlaybackSpeed: (speed: number) => void;
  toggleVelocity: () => void;
  toggleTrails: () => void;
  toggleIds: () => void;
  toggleSidebar: () => void;
  toggleLayer: (key: LayerKey) => void;
  selectPlayer: (id: number, multi?: boolean) => void;
  clearSelectedPlayers: () => void;
  toggleBottomPanel: () => void;
  setBottomPanelHeight: (h: number) => void;
}

const defaultLayers: Record<LayerKey, boolean> = {
  voronoi: false, heatmap: false, passNetwork: false, pressure: false, coverShadow: false,
  formation: false, teamCentroid: false, teamWidthDepth: false, compactness: false,
  shotMap: false, passSonar: false, zone14: false, buildupSequence: false,
  duelHeatmap: false, transitions: false, setPieces: false,
};

export const usePlaybackStore = create<PlaybackState>((set) => ({
  trackingData: null,
  currentFrame: 0,
  isPlaying: false,
  playbackSpeed: 1,
  showVelocity: false,
  showTrails: false,
  showIds: true,

  sidebarOpen: true,
  layers: { ...defaultLayers },
  selectedPlayerIds: [],

  bottomPanelOpen: false,
  bottomPanelHeight: 280,

  setTrackingData: (data) => set({ trackingData: data, currentFrame: 0 }),
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  togglePlaying: () => set((s) => ({ isPlaying: !s.isPlaying })),
  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),
  toggleVelocity: () => set((s) => ({ showVelocity: !s.showVelocity })),
  toggleTrails: () => set((s) => ({ showTrails: !s.showTrails })),
  toggleIds: () => set((s) => ({ showIds: !s.showIds })),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  toggleLayer: (key) => set((s) => ({ layers: { ...s.layers, [key]: !s.layers[key] } })),
  selectPlayer: (id, multi) =>
    set((s) => {
      if (multi) {
        const ids = s.selectedPlayerIds.includes(id)
          ? s.selectedPlayerIds.filter((x) => x !== id)
          : [...s.selectedPlayerIds, id];
        return { selectedPlayerIds: ids };
      }
      return { selectedPlayerIds: s.selectedPlayerIds.includes(id) && s.selectedPlayerIds.length === 1 ? [] : [id] };
    }),
  clearSelectedPlayers: () => set({ selectedPlayerIds: [] }),
  toggleBottomPanel: () => set((s) => ({ bottomPanelOpen: !s.bottomPanelOpen })),
  setBottomPanelHeight: (h) => set({ bottomPanelHeight: h }),
}));
