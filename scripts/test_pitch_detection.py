"""
Standalone pitch keypoint detection test.
Runs AIPitchEstimator on a video and produces a visualised output
showing which keypoints are detected per frame and at what confidence.

Usage:
    uv run python test_pitch_detection.py
"""

import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from tactix.config import Config
from tactix.vision.calibration.ai_estimator import AIPitchEstimator
from tactix.core.keypoints import YOLO_INDEX_MAP

# ── Config ────────────────────────────────────────────────────────────────────
cfg = Config()
INPUT_VIDEO  = "assets/samples/test1.mp4"
OUTPUT_VIDEO = "assets/output/test_pitch_detection.mp4"
CONF_THRESH  = cfg.CONF_PITCH   # 0.3 — same threshold the engine uses
FRAME_SKIP   = 1                # process every frame (set >1 to go faster)

# ── Colour coding (BGR) ───────────────────────────────────────────────────────
def conf_colour(c: float):
    if c >= 0.6:
        return (0, 220, 0)      # green  — high confidence
    elif c >= CONF_THRESH:
        return (0, 200, 255)    # yellow — above threshold
    else:
        return (0, 0, 200)      # red    — below threshold (not used by engine)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model: {cfg.PITCH_MODEL_PATH}")
estimator = AIPitchEstimator(cfg.PITCH_MODEL_PATH, cfg.DEVICE)

# ── Open video ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS) or 30
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

print(f"Input : {INPUT_VIDEO}  ({total_frames} frames, {w}x{h} @ {fps:.1f} fps)")
print(f"Output: {OUTPUT_VIDEO}")
print(f"Threshold: {CONF_THRESH}")
print()

# ── Stats accumulators ────────────────────────────────────────────────────────
frames_processed  = 0
frames_with_any   = 0
frames_above_thr  = 0          # frames with >= 4 keypoints above threshold
kpt_conf_sum      = defaultdict(float)   # sum of conf per keypoint index
kpt_detect_count  = defaultdict(int)     # frames where kpt conf >= threshold
above_thr_counts  = []                   # n keypoints >= threshold per frame

# ── Main loop ─────────────────────────────────────────────────────────────────
for frame_idx in tqdm(range(total_frames), desc="Processing"):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_SKIP != 0:
        out.write(frame)
        continue

    frames_processed += 1
    xy, conf = estimator.predict(frame)

    # ── Draw ──────────────────────────────────────────────────────────────────
    canvas = frame.copy()

    if xy is not None:
        frames_with_any += 1
        n_above = 0

        for i, ((x, y), c) in enumerate(zip(xy, conf)):
            # Always draw every predicted point (colour shows confidence tier)
            colour = conf_colour(c)
            ix, iy = int(x), int(y)

            # Skip (0, 0) placeholder predictions
            if ix == 0 and iy == 0:
                continue

            # Circle: filled if above threshold, outline if below
            if c >= CONF_THRESH:
                cv2.circle(canvas, (ix, iy), 5, colour, -1)
                n_above += 1
                kpt_detect_count[i] += 1
            else:
                cv2.circle(canvas, (ix, iy), 4, colour, 1)

            # Label: name + conf
            name  = YOLO_INDEX_MAP.get(i, f"kpt{i}")
            short = name.replace("_CORNER", "").replace("_LINE", "").replace("_CORNER", "")
            label = f"{short} {c:.2f}"
            cv2.putText(canvas, label, (ix + 5, iy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)

            kpt_conf_sum[i] += c

        above_thr_counts.append(n_above)
        if n_above >= 4:
            frames_above_thr += 1

        # HUD
        hud = f"Frame {frame_idx}/{total_frames}  |  detected (>={CONF_THRESH}): {n_above}/26"
    else:
        above_thr_counts.append(0)
        hud = f"Frame {frame_idx}/{total_frames}  |  NO DETECTION"

    cv2.putText(canvas, hud, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend
    for label, colour, y0 in [
        (">=0.6 high conf", (0, 220, 0),   h - 55),
        (">=0.3 above thr", (0, 200, 255), h - 35),
        ("<0.3  below thr", (0, 0, 200),   h - 15),
    ]:
        cv2.circle(canvas, (12, y0), 5, colour, -1)
        cv2.putText(canvas, label, (22, y0 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)

    out.write(canvas)

cap.release()
out.release()

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("═" * 60)
print(f"SUMMARY  ({frames_processed} frames processed out of {total_frames})")
print("═" * 60)
print(f"  Frames with any detection    : {frames_with_any}/{frames_processed}")
print(f"  Frames with >= 4 kpts (thr)  : {frames_above_thr}/{frames_processed}  "
      f"({100*frames_above_thr/max(frames_processed,1):.1f}%)")
if above_thr_counts:
    print(f"  Avg kpts above threshold/frame: {np.mean(above_thr_counts):.1f}  "
          f"(min {min(above_thr_counts)}, max {max(above_thr_counts)})")

print()
print("  Per-keypoint detection rate (frames where conf >= threshold):")
print(f"  {'Idx':>3}  {'Name':<25}  {'Detected':>10}  {'Rate':>7}  {'AvgConf':>8}")
print(f"  {'---':>3}  {'----':<25}  {'--------':>10}  {'----':>7}  {'-------':>8}")
for i in range(26):
    name    = YOLO_INDEX_MAP.get(i, f"kpt{i}")
    det     = kpt_detect_count[i]
    rate    = det / max(frames_processed, 1)
    avg_c   = kpt_conf_sum[i] / det if det > 0 else 0.0
    flag    = "  ← low" if rate < 0.3 else ""
    print(f"  {i:>3}  {name:<25}  {det:>10}  {rate:>6.1%}  {avg_c:>8.3f}{flag}")

print()
print(f"Output saved to: {OUTPUT_VIDEO}")
