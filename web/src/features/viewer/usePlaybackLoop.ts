import { useEffect, useRef, useCallback } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";

/**
 * Drives the playback loop: advances currentFrame based on playbackSpeed and fps.
 * Also syncs an optional <video> element to the same time.
 */
export function usePlaybackLoop(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const rafId = useRef<number>(0);
  const lastTs = useRef<number>(0);
  const accumulator = useRef<number>(0);

  const tick = useCallback((ts: number) => {
    const { isPlaying, playbackSpeed, currentFrame, trackingData, setCurrentFrame } =
      usePlaybackStore.getState();

    if (!isPlaying || !trackingData) {
      lastTs.current = ts;
      rafId.current = requestAnimationFrame(tick);
      return;
    }

    const dt = lastTs.current ? (ts - lastTs.current) / 1000 : 0;
    lastTs.current = ts;

    const fps = trackingData.meta.fps;
    const totalFrames = trackingData.meta.total_frames;
    const frameDuration = 1 / fps;

    accumulator.current += dt * playbackSpeed;

    if (accumulator.current >= frameDuration) {
      const framesToAdvance = Math.floor(accumulator.current / frameDuration);
      accumulator.current -= framesToAdvance * frameDuration;

      const newFrame = Math.min(currentFrame + framesToAdvance, totalFrames - 1);
      setCurrentFrame(newFrame);

      // Sync video element
      if (videoRef.current) {
        const targetTime = newFrame / fps;
        // Only seek if drift > half a frame
        if (Math.abs(videoRef.current.currentTime - targetTime) > frameDuration * 0.5) {
          videoRef.current.currentTime = targetTime;
        }
      }

      if (newFrame >= totalFrames - 1) {
        usePlaybackStore.getState().setIsPlaying(false);
      }
    }

    rafId.current = requestAnimationFrame(tick);
  }, [videoRef]);

  useEffect(() => {
    rafId.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId.current);
  }, [tick]);

  // Sync video when frame changes manually (scrubbing)
  useEffect(() => {
    const unsub = usePlaybackStore.subscribe((state, prev) => {
      if (state.currentFrame !== prev.currentFrame && !state.isPlaying && videoRef.current) {
        const fps = state.trackingData?.meta.fps ?? 25;
        videoRef.current.currentTime = state.currentFrame / fps;
      }
    });
    return unsub;
  }, [videoRef]);
}
