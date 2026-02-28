"""
Convert 48-point (v15i) pitch keypoint labels to 26-point format.

Reads YOLO-Pose label files from the v15i dataset, extracts the 26 keypoints
that correspond to the v5i (26-point) dataset, remaps them to the correct
indices, and writes new label files.

Usage:
    python training/scripts/convert_v15i_to_26pt.py \
        --input "datasets/Football Pitch Keypoints.v15i.yolov8" \
        --output datasets/v15i_converted_26pt
"""

import argparse
import shutil
from pathlib import Path


# 48-point (0-indexed) -> 26-point (0-indexed) mapping
# Derived from spatial layout comparison (user-provided, 1-indexed, converted to 0-indexed)
MAPPING_48_TO_26: dict[int, int] = {
    # Left side (goal line x=0 and inner corners)
    0: 0,    # TL_CORNER
    1: 3,    # L_PA_TOP_LINE
    2: 4,    # L_GA_TOP_LINE
    3: 5,    # L_GA_BOTTOM_LINE
    4: 6,    # L_PA_BOTTOM_LINE
    5: 7,    # BL_CORNER
    6: 18,   # L_GA_TOP_CORNER
    7: 19,   # L_GA_BOTTOM_CORNER
    9: 16,   # L_PA_TOP_CORNER
    12: 17,  # L_PA_BOTTOM_CORNER
    # Center
    13: 23,  # CIRCLE_LEFT
    14: 1,   # MID_TOP
    15: 22,  # CIRCLE_TOP
    16: 24,  # CENTER_SPOT
    17: 8,   # MID_BOTTOM
    18: 25,  # CIRCLE_RIGHT
    # Right side (inner corners and goal line x=105)
    19: 14,  # R_PA_TOP_CORNER
    22: 15,  # R_PA_BOTTOM_CORNER
    24: 20,  # R_GA_TOP_CORNER
    25: 21,  # R_GA_BOTTOM_CORNER
    26: 2,   # TR_CORNER
    27: 13,  # R_PA_TOP_LINE
    28: 12,  # R_GA_TOP_LINE
    29: 11,  # R_GA_BOTTOM_LINE
    30: 10,  # R_PA_BOTTOM_LINE
    31: 9,   # BR_CORNER
}

NUM_OLD_KPT = 48
NUM_NEW_KPT = 26


def convert_label_line(line: str) -> str | None:
    """Convert a single YOLO-Pose label line from 48-point to 26-point format."""
    parts = line.strip().split()
    if not parts:
        return None

    # bbox: class_id cx cy w h (5 values)
    expected_len = 5 + NUM_OLD_KPT * 3
    if len(parts) != expected_len:
        print(f"  Warning: expected {expected_len} fields, got {len(parts)}, skipping line")
        return None

    bbox = parts[:5]

    # Parse 48 keypoints as (x, y, v) triplets
    kpt_data = parts[5:]
    old_kpts: list[tuple[str, str, str]] = []
    for i in range(NUM_OLD_KPT):
        x = kpt_data[i * 3]
        y = kpt_data[i * 3 + 1]
        v = kpt_data[i * 3 + 2]
        old_kpts.append((x, y, v))

    # Build 26-point output (initialize all as invisible/unlabeled)
    new_kpts: list[tuple[str, str, str]] = [("0", "0", "0")] * NUM_NEW_KPT

    for old_idx, new_idx in MAPPING_48_TO_26.items():
        new_kpts[new_idx] = old_kpts[old_idx]

    # Reconstruct label line
    kpt_str = " ".join(f"{x} {y} {v}" for x, y, v in new_kpts)
    return f"{' '.join(bbox)} {kpt_str}"


def convert_split(input_dir: Path, output_dir: Path, split: str) -> int:
    """Convert all label files in a dataset split."""
    # v15i uses 'valid' not 'val'
    input_labels = input_dir / split / "labels"
    input_images = input_dir / split / "images"
    output_labels = output_dir / split / "labels"
    output_images = output_dir / split / "images"

    if not input_labels.exists():
        print(f"  Split '{split}' not found at {input_labels}, skipping")
        return 0

    output_labels.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    count = 0
    for label_file in sorted(input_labels.glob("*.txt")):
        lines_out = []
        for line in label_file.read_text().strip().splitlines():
            converted = convert_label_line(line)
            if converted:
                lines_out.append(converted)

        if lines_out:
            (output_labels / label_file.name).write_text("\n".join(lines_out) + "\n")
            count += 1

        # Copy corresponding image
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            img_file = input_images / (label_file.stem + ext)
            if img_file.exists():
                shutil.copy2(img_file, output_images / img_file.name)
                break

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert v15i 48-point labels to 26-point format")
    parser.add_argument("--input", type=str,
                        default="datasets/Football Pitch Keypoints.v15i.yolov8",
                        help="Path to v15i dataset")
    parser.add_argument("--output", type=str,
                        default="datasets/v15i_converted_26pt",
                        help="Output directory for converted dataset")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"Converting: {input_dir} -> {output_dir}")
    print(f"Mapping {NUM_OLD_KPT} keypoints -> {NUM_NEW_KPT} keypoints")
    print(f"Using {len(MAPPING_48_TO_26)} mapped points, discarding {NUM_OLD_KPT - len(MAPPING_48_TO_26)} points\n")

    total = 0
    for split in ["train", "valid", "test"]:
        count = convert_split(input_dir, output_dir, split)
        print(f"  {split}: {count} labels converted")
        total += count

    print(f"\nTotal: {total} labels converted")

    # Verify a sample
    sample_labels = list((output_dir / "train" / "labels").glob("*.txt"))
    if sample_labels:
        sample = sample_labels[0].read_text().strip().split()
        expected = 5 + NUM_NEW_KPT * 3
        print(f"\nVerification: {sample_labels[0].name} has {len(sample)} fields (expected {expected})")
        if len(sample) == expected:
            print("  OK")
        else:
            print("  MISMATCH!")


if __name__ == "__main__":
    main()
