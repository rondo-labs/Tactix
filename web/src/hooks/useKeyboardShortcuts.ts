import { useEffect } from "react";
import { usePlaybackStore } from "../stores/playbackStore";

export function useKeyboardShortcuts() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      const state = usePlaybackStore.getState();
      const totalFrames = state.trackingData?.meta.total_frames ?? 0;

      switch (e.key) {
        case " ":
          e.preventDefault();
          state.togglePlaying();
          break;
        case "ArrowLeft":
          e.preventDefault();
          state.setCurrentFrame(Math.max(0, state.currentFrame - (e.shiftKey ? 10 : 1)));
          break;
        case "ArrowRight":
          e.preventDefault();
          state.setCurrentFrame(Math.min(totalFrames - 1, state.currentFrame + (e.shiftKey ? 10 : 1)));
          break;
        case "ArrowUp":
          e.preventDefault();
          {
            const speeds = [0.25, 0.5, 1, 2, 4];
            const idx = speeds.indexOf(state.playbackSpeed);
            if (idx < speeds.length - 1) state.setPlaybackSpeed(speeds[idx + 1]);
          }
          break;
        case "ArrowDown":
          e.preventDefault();
          {
            const speeds = [0.25, 0.5, 1, 2, 4];
            const idx = speeds.indexOf(state.playbackSpeed);
            if (idx > 0) state.setPlaybackSpeed(speeds[idx - 1]);
          }
          break;
        case "v":
          state.toggleVelocity();
          break;
        case "i":
          state.toggleIds();
          break;
        case "s":
          state.toggleSidebar();
          break;
        case "b":
          state.toggleBottomPanel();
          break;
        case "Escape":
          state.clearSelectedPlayers();
          break;
        case "Home":
          e.preventDefault();
          state.setCurrentFrame(0);
          break;
        case "End":
          e.preventDefault();
          state.setCurrentFrame(totalFrames - 1);
          break;
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
}
