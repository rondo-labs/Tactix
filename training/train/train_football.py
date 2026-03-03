"""
RF-DETR Football Player Detection - Training Script
====================================================

Prerequisites (run once):
    pip install rfdetr

Then run:
    python train_football.py

Trained weights will be saved to:
    runs/rfdetr/checkpoint_best_total.pth
"""

import os
import shutil

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these if needed
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIR  = "football-players-detection.v20-rf-detr-m.yolo26"  # folder next to this script
OUTPUT_DIR   = "runs/rfdetr"
EPOCHS       = 50
BATCH_SIZE   = 8       # RTX 4090 can handle 8 or 16 comfortably
GRAD_ACCUM   = 2       # effective batch = BATCH_SIZE * GRAD_ACCUM
LR           = 1e-4
DEVICE       = "cuda"  # RTX 4090
MODEL_SIZE   = "large" # "large" (better) or "base" (faster)
CLASS_NAMES  = ["ball", "goalkeeper", "player", "referee"]
# ──────────────────────────────────────────────────────────────────────────────


def main():
    try:
        from rfdetr import RFDETRBase, RFDETRLarge
    except ImportError:
        print("ERROR: rfdetr is not installed.")
        print("Run:  pip install rfdetr")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 55)
    print("  RF-DETR  —  Football Player Detection")
    print("=" * 55)
    print(f"  Model   : rfdetr-{MODEL_SIZE}")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Batch   : {BATCH_SIZE}  (×{GRAD_ACCUM} grad accum = {BATCH_SIZE * GRAD_ACCUM} effective)")
    print(f"  LR      : {LR}")
    print(f"  Device  : {DEVICE}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 55 + "\n")

    ModelClass = RFDETRLarge if MODEL_SIZE == "large" else RFDETRBase
    model = ModelClass()

    model.train(
        dataset_dir=DATASET_DIR,
        dataset_file="yolo",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM,
        lr=LR,
        lr_encoder=LR * 1.5,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        class_names=CLASS_NAMES,
        multi_scale=True,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=10,
        checkpoint_interval=5,
        progress_bar=True,
    )

    # Find best checkpoint
    best = os.path.join(OUTPUT_DIR, "checkpoint_best_total.pth")
    if not os.path.exists(best):
        candidates = sorted(
            [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pth")],
            key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
        )
        best = os.path.join(OUTPUT_DIR, candidates[-1]) if candidates else None

    if best and os.path.exists(best):
        dest = f"rfdetr_{MODEL_SIZE}_football_best.pth"
        shutil.copy2(best, dest)
        print(f"\n✅ Done! Best weights saved to: {dest}")
        print(f"   Send this file back to use in Tactix.")
    else:
        print(f"\n⚠️  Could not find best checkpoint in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
