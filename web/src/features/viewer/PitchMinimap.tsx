import { useRef, useEffect, useMemo } from "react";
import { usePlaybackStore } from "../../stores/playbackStore";
import type { TrackingFrame } from "../../types/tracking";
import { computeVoronoi } from "../../lib/computeVoronoi";
import { computeHeatmap, renderHeatmapToCanvas } from "../../lib/computeHeatmap";

const TEAM_COLORS: Record<string, string> = {
  A: "#e63946",
  B: "#457b9d",
  REFEREE: "#ffd60a", REF: "#ffd60a",
  GOALKEEPER: "#a8dadc", GK: "#a8dadc",
  UNKNOWN: "#888888", "?": "#888888",
};

const PITCH = { L: 105, W: 68 };

interface Props {
  width: number;
  height: number;
  frame: TrackingFrame | null;
}

export default function PitchMinimap({ width, height, frame }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { showVelocity, showIds, selectedPlayerIds, layers } = usePlaybackStore();
  const meta = usePlaybackStore((s) => s.trackingData?.meta);
  const trackingData = usePlaybackStore((s) => s.trackingData);
  const currentFrame = usePlaybackStore((s) => s.currentFrame);

  // Throttled heatmap: only recompute every 5 frames
  const heatmapBucket = Math.floor(currentFrame / 5) * 5;
  const heatmapGrid = useMemo(() => {
    if (!layers.heatmap || !trackingData) return null;
    const srcW = meta?.canvas.width ?? 1559;
    const srcH = meta?.canvas.height ?? 1010;
    return computeHeatmap(
      trackingData.frames, heatmapBucket, 40, 26,
      srcW, srcH,
      selectedPlayerIds.length > 0 ? selectedPlayerIds : undefined,
    );
  }, [layers.heatmap, trackingData, heatmapBucket, meta, selectedPlayerIds]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0 || height <= 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const margin = width * 0.018;
    const pw = width - margin * 2;
    const ph = height - margin * 2;

    const mx = (m: number) => margin + (m / PITCH.L) * pw;
    const my = (m: number) => margin + (m / PITCH.W) * ph;
    const ms = (m: number) => (m / PITCH.L) * pw;
    const msv = (m: number) => (m / PITCH.W) * ph;

    const srcW = meta?.canvas.width ?? 1559;
    const srcH = meta?.canvas.height ?? 1010;
    const sx = (v: number) => margin + (v / srcW) * pw;
    const sy = (v: number) => margin + (v / srcH) * ph;

    const lineWidth = Math.max(1, pw * 0.0013);

    // ── Clear ──
    ctx.clearRect(0, 0, width, height);

    // ── Background ──
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, width, height);

    // Grass stripes
    const stripeCount = 10;
    const stripeW = pw / stripeCount;
    for (let i = 0; i < stripeCount; i++) {
      ctx.fillStyle = i % 2 === 0 ? "#0e1e14" : "#0a1710";
      ctx.fillRect(margin + i * stripeW, margin, stripeW, ph);
    }

    // ── Heatmap overlay ──
    if (heatmapGrid) {
      const imgData = renderHeatmapToCanvas(heatmapGrid, 40, 26, width, height, margin, margin, pw, ph);
      const offscreen = new OffscreenCanvas(width, height);
      const offCtx = offscreen.getContext("2d")!;
      offCtx.putImageData(imgData, 0, 0);
      ctx.drawImage(offscreen, 0, 0);
    }

    // ── Voronoi overlay ──
    if (layers.voronoi && frame?.players) {
      const voronoi = computeVoronoi(frame.players, pw, ph, sx, sy, srcW, srcH);
      for (const cell of voronoi.cellPaths) {
        ctx.fillStyle = cell.team === "A" ? "rgba(230,57,70,0.10)" : "rgba(69,123,157,0.10)";
        ctx.fill(cell.path);
        ctx.strokeStyle = cell.team === "A" ? "rgba(230,57,70,0.25)" : "rgba(69,123,157,0.25)";
        ctx.lineWidth = 0.8;
        ctx.stroke(cell.path);
      }
    }

    // ── Pitch lines ──
    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";

    ctx.strokeRect(mx(0), my(0), ms(PITCH.L), msv(PITCH.W));

    ctx.beginPath();
    ctx.moveTo(mx(52.5), my(0));
    ctx.lineTo(mx(52.5), my(PITCH.W));
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(mx(52.5), my(34), ms(9.15), 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.beginPath();
    ctx.arc(mx(52.5), my(34), Math.max(2, ms(0.35)), 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(mx(0), my((PITCH.W - 40.32) / 2), ms(16.5), msv(40.32));
    ctx.strokeRect(mx(PITCH.L - 16.5), my((PITCH.W - 40.32) / 2), ms(16.5), msv(40.32));

    ctx.strokeRect(mx(0), my((PITCH.W - 18.32) / 2), ms(5.5), msv(18.32));
    ctx.strokeRect(mx(PITCH.L - 5.5), my((PITCH.W - 18.32) / 2), ms(5.5), msv(18.32));

    ctx.fillStyle = "rgba(255,255,255,0.4)";
    for (const xm of [11, PITCH.L - 11]) {
      ctx.beginPath();
      ctx.arc(mx(xm), my(34), Math.max(2, ms(0.35)), 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = lineWidth;
    const arcAngle = Math.acos(5.5 / 9.15);
    ctx.beginPath();
    ctx.arc(mx(11), my(34), ms(9.15), -arcAngle, arcAngle);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(mx(PITCH.L - 11), my(34), ms(9.15), Math.PI - arcAngle, Math.PI + arcAngle);
    ctx.stroke();

    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = lineWidth * 1.5;
    ctx.fillStyle = "rgba(255,255,255,0.05)";
    ctx.fillRect(mx(-1.5), my((PITCH.W - 7.32) / 2), ms(1.5), msv(7.32));
    ctx.strokeRect(mx(-1.5), my((PITCH.W - 7.32) / 2), ms(1.5), msv(7.32));
    ctx.fillRect(mx(PITCH.L), my((PITCH.W - 7.32) / 2), ms(1.5), msv(7.32));
    ctx.strokeRect(mx(PITCH.L), my((PITCH.W - 7.32) / 2), ms(1.5), msv(7.32));

    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = lineWidth;
    for (const [cx, cy, sa, ea] of [
      [mx(0), my(0), 0, Math.PI / 2],
      [mx(PITCH.L), my(0), Math.PI / 2, Math.PI],
      [mx(PITCH.L), my(PITCH.W), Math.PI, Math.PI * 1.5],
      [mx(0), my(PITCH.W), Math.PI * 1.5, Math.PI * 2],
    ] as [number, number, number, number][]) {
      ctx.beginPath();
      ctx.arc(cx, cy, ms(1), sa, ea);
      ctx.stroke();
    }

    // ── Team centroid ──
    if (layers.teamCentroid && frame?.players) {
      for (const team of ["A", "B"] as const) {
        const tp = frame.players.filter((p) => p.team === team);
        if (tp.length === 0) continue;
        const cx = tp.reduce((s, p) => s + sx(p.x), 0) / tp.length;
        const cy = tp.reduce((s, p) => s + sy(p.y), 0) / tp.length;
        const color = TEAM_COLORS[team];
        // Cross marker
        const sz = 8;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = 0.7;
        ctx.beginPath(); ctx.moveTo(cx - sz, cy); ctx.lineTo(cx + sz, cy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx, cy - sz); ctx.lineTo(cx, cy + sz); ctx.stroke();
        // Circle around centroid
        ctx.beginPath();
        ctx.arc(cx, cy, sz + 2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // ── Players ──
    if (frame?.players) {
      const r = Math.max(5, pw * 0.007);
      const hasSelection = selectedPlayerIds.length > 0;

      for (const p of frame.players) {
        const px = sx(p.x);
        const py = sy(p.y);
        const color = TEAM_COLORS[p.team] ?? TEAM_COLORS.UNKNOWN;
        const isSelected = selectedPlayerIds.includes(p.id);
        const dimmed = hasSelection && !isSelected;

        ctx.globalAlpha = dimmed ? 0.25 : 1;

        // Selection ring
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(px, py, r + 4, 0, Math.PI * 2);
          ctx.strokeStyle = "#58a6ff";
          ctx.lineWidth = 2;
          ctx.stroke();
          // Pulse glow
          ctx.beginPath();
          ctx.arc(px, py, r + 6, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(88,166,255,0.3)";
          ctx.lineWidth = 3;
          ctx.stroke();
        }

        // Dot
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // ID label
        if (showIds) {
          ctx.fillStyle = "white";
          ctx.font = `${Math.max(8, r * 1.2)}px sans-serif`;
          ctx.textAlign = "center";
          ctx.fillText(p.jersey ?? String(p.id), px, py - r - 3);
        }

        // Velocity arrow
        if (showVelocity && p.vx != null && p.vy != null && (p.speed ?? 0) > 0.5) {
          const ex = px + p.vx * 8;
          const ey = py + p.vy * 8;
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(ex, ey);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = dimmed ? 0.15 : 0.8;
          ctx.stroke();
          const angle = Math.atan2(ey - py, ex - px);
          ctx.beginPath();
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex - 5 * Math.cos(angle - 0.4), ey - 5 * Math.sin(angle - 0.4));
          ctx.lineTo(ex - 5 * Math.cos(angle + 0.4), ey - 5 * Math.sin(angle + 0.4));
          ctx.closePath();
          ctx.fillStyle = color;
          ctx.fill();
        }

        ctx.globalAlpha = 1;
      }
    }

    // ── Ball ──
    if (frame?.ball) {
      const bx = sx(frame.ball.x);
      const by = sy(frame.ball.y);
      ctx.beginPath();
      ctx.arc(bx, by, Math.max(6, pw * 0.007), 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,165,0,0.3)";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(bx, by, Math.max(4, pw * 0.005), 0, Math.PI * 2);
      ctx.fillStyle = "#ffa500";
      ctx.fill();
    }
  }, [width, height, frame, showVelocity, showIds, meta, selectedPlayerIds, layers, heatmapGrid]);

  if (width <= 0 || height <= 0) return null;

  return <canvas ref={canvasRef} width={width} height={height} style={{ borderRadius: 6 }} />;
}
