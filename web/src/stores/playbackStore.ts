import { create } from "zustand";
import type { TrackingData } from "../types/tracking";

interface PlaybackState {
  trackingData: TrackingData | null;
  currentFrame: number;
  isPlaying: boolean;
  playbackSpeed: number;
  showVelocity: boolean;
  showTrails: boolean;
  showIds: boolean;

  setTrackingData: (data: TrackingData) => void;
  setCurrentFrame: (frame: number) => void;
  setIsPlaying: (playing: boolean) => void;
  togglePlaying: () => void;
  setPlaybackSpeed: (speed: number) => void;
  toggleVelocity: () => void;
  toggleTrails: () => void;
  toggleIds: () => void;
}

export const usePlaybackStore = create<PlaybackState>((set) => ({
  trackingData: null,
  currentFrame: 0,
  isPlaying: false,
  playbackSpeed: 1,
  showVelocity: false,
  showTrails: false,
  showIds: true,

  setTrackingData: (data) => set({ trackingData: data, currentFrame: 0 }),
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  togglePlaying: () => set((s) => ({ isPlaying: !s.isPlaying })),
  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),
  toggleVelocity: () => set((s) => ({ showVelocity: !s.showVelocity })),
  toggleTrails: () => set((s) => ({ showTrails: !s.showTrails })),
  toggleIds: () => set((s) => ({ showIds: !s.showIds })),
}));
