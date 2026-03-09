import type { TrackingFrame } from "../types/tracking";

export function computeHeatmap(
  frames: TrackingFrame[],
  currentFrame: number,
  gridW: number,
  gridH: number,
  srcW: number,
  srcH: number,
  playerIds?: number[],
  windowSize = 150,
): Float32Array {
  const grid = new Float32Array(gridW * gridH);
  const start = Math.max(0, currentFrame - windowSize);
  const end = currentFrame;

  for (let f = start; f <= end && f < frames.length; f++) {
    const weight = 1 - (end - f) / windowSize;
    for (const p of frames[f].players) {
      if (playerIds && playerIds.length > 0 && !playerIds.includes(p.id)) continue;
      if (p.team !== "A" && p.team !== "B") continue;

      const gx = Math.floor((p.x / srcW) * gridW);
      const gy = Math.floor((p.y / srcH) * gridH);
      if (gx < 0 || gx >= gridW || gy < 0 || gy >= gridH) continue;
      grid[gy * gridW + gx] += weight;
    }
  }

  // Gaussian blur (simple 3x3 box blur, 2 passes)
  const tmp = new Float32Array(gridW * gridH);
  for (let pass = 0; pass < 2; pass++) {
    const src = pass === 0 ? grid : tmp;
    const dst = pass === 0 ? tmp : grid;
    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        let sum = 0;
        let count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < gridW && ny >= 0 && ny < gridH) {
              sum += src[ny * gridW + nx];
              count++;
            }
          }
        }
        dst[y * gridW + x] = sum / count;
      }
    }
  }

  return grid;
}

export function renderHeatmapToCanvas(
  grid: Float32Array,
  gridW: number,
  gridH: number,
  canvasW: number,
  canvasH: number,
  marginX: number,
  marginY: number,
  pitchW: number,
  pitchH: number,
): ImageData {
  const imgData = new ImageData(canvasW, canvasH);
  const data = imgData.data;

  let maxVal = 0;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > maxVal) maxVal = grid[i];
  }
  if (maxVal === 0) return imgData;

  const cellW = pitchW / gridW;
  const cellH = pitchH / gridH;

  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const val = grid[gy * gridW + gx] / maxVal;
      if (val < 0.01) continue;

      const px0 = Math.floor(marginX + gx * cellW);
      const py0 = Math.floor(marginY + gy * cellH);
      const px1 = Math.ceil(marginX + (gx + 1) * cellW);
      const py1 = Math.ceil(marginY + (gy + 1) * cellH);

      // Color: blue → cyan → green → yellow → red
      let r: number, g: number, b: number;
      if (val < 0.25) { const t = val / 0.25; r = 0; g = Math.floor(t * 200); b = 200; }
      else if (val < 0.5) { const t = (val - 0.25) / 0.25; r = 0; g = 200; b = Math.floor(200 * (1 - t)); }
      else if (val < 0.75) { const t = (val - 0.5) / 0.25; r = Math.floor(t * 255); g = 200; b = 0; }
      else { const t = (val - 0.75) / 0.25; r = 255; g = Math.floor(200 * (1 - t)); b = 0; }

      const alpha = Math.floor(val * 140);

      for (let py = py0; py < py1 && py < canvasH; py++) {
        for (let px = px0; px < px1 && px < canvasW; px++) {
          const idx = (py * canvasW + px) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          data[idx + 3] = alpha;
        }
      }
    }
  }

  return imgData;
}
