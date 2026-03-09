"""
Project: Tactix
File Created: 2026-03-03
File Name: train_rfdetr.py
Description:
    Fine-tune RF-DETR on the football players detection dataset.

    IMPORTANT — run this script via uv to avoid conflicting with the main
    project's transformers 5.x requirement:

        uv run --with rfdetr --with "transformers<5.0" \\
            python training/train/train_rfdetr.py

    Trained weights are saved to:
        assets/weights/rfdetr_football_best.pth
"""

import argparse
import os
import shutil
import sys

# Resolve project root regardless of where the script is invoked from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "assets", "weights")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training", "train", "runs", "rfdetr")

CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR for Tactix")
    parser.add_argument(
        "--dataset",
        default=os.path.join(PROJECT_ROOT, "datasets", "football-players-detection.v20-rf-detr-m.yolo26"),
        help="Path to YOLO-format dataset directory containing data.yaml",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--model",
        choices=["large", "base"],
        default="large",
        help="RF-DETR model size. 'large' = higher accuracy, 'base' = faster",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from rfdetr import RFDETRBase, RFDETRLarge
    except ImportError:
        print(
            "ERROR: rfdetr is not installed in this environment.\n"
            "Run this script with:\n"
            "  uv run --with rfdetr --with 'transformers<5.0' python training/train/train_rfdetr.py"
        )
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  RF-DETR Fine-Tune — Tactix Football Detection")
    print(f"{'=' * 60}")
    print(f"  Model   : rfdetr-{args.model}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}  (grad_accum=4 → effective={args.batch_size * 4})")
    print(f"  LR      : {args.lr}")
    print(f"  Device  : {args.device}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    # Choose model class
    ModelClass = RFDETRLarge if args.model == "large" else RFDETRBase
    model = ModelClass()

    model.train(
        dataset_dir=args.dataset,
        dataset_file="yolo",          # YOLO-format labels in train/labels/ and valid/labels/
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=4,           # effective batch = batch_size * 4
        lr=args.lr,
        lr_encoder=args.lr * 1.5,
        device=args.device,
        output_dir=OUTPUT_DIR,
        class_names=CLASS_NAMES,
        multi_scale=True,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=10,
        checkpoint_interval=10,
        progress_bar=True,
        resume=args.resume,
    )

    # Copy best checkpoint to assets/weights for use in the pipeline
    best_src = os.path.join(OUTPUT_DIR, "checkpoint_best_total.pth")
    if not os.path.exists(best_src):
        # Fallback: grab the last checkpoint
        candidates = sorted(
            [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pth")],
            key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
        )
        best_src = os.path.join(OUTPUT_DIR, candidates[-1]) if candidates else None

    if best_src and os.path.exists(best_src):
        dest = os.path.join(WEIGHTS_DIR, f"rfdetr_{args.model}_football_best.pth")
        shutil.copy2(best_src, dest)
        print(f"\n✅ Best weights copied to: {dest}")
        print(f"   Set in config.py:  RFDETR_MODEL_PATH = \"{dest}\"")
    else:
        print(f"\n⚠️  Could not locate best checkpoint in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
