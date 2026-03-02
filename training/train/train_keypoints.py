"""
Project: Tactix
File Created: 2026-02-07 18:00:00
Author: Xingnan Zhu
File Name: train_keypoints.py
Description:
    Standard script to train a YOLO26-Pose model (Pitch Keypoints).
    Usage: python training/train/train_keypoints.py
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def resolve_data_yaml(yaml_path: str) -> str:
    """
    Ultralytics resolves relative 'path' values in dataset YAMLs using its
    global settings.datasets_dir, which may point to a different project.

    This function reads the YAML, converts a relative 'path' to an absolute
    path anchored at the YAML file's own directory (the natural convention),
    and writes a temporary YAML that Ultralytics will find correctly.

    Works for both:
      - training/configs/pitch_kpt_26.yaml  (path: ../datasets/pitch_keypoints_26pt)
      - datasets/pitch_keypoints_26pt/data.yaml  (path: .)

    Returns the path to the resolved YAML (caller must delete tmp file).
    """
    yaml_dir = Path(yaml_path).parent.resolve()

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg.get("path", "")
    if dataset_path and not Path(dataset_path).is_absolute():
        abs_dataset = (yaml_dir / dataset_path).resolve()
        cfg["path"] = str(abs_dataset)
        print(f"📂 Dataset path resolved: {abs_dataset}")

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
            dir=os.path.dirname(yaml_path),
        )
        yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
        tmp.close()
        return tmp.name

    return yaml_path

def get_weights_path(model_name):
    """
    Resolves the path to the model weights.
    Checks 'assets/weights/' first. If not found, returns the name directly.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    weights_dir = os.path.join(project_root, "assets/weights")
    
    os.makedirs(weights_dir, exist_ok=True)
    
    target_path = os.path.join(weights_dir, model_name)
    
    if os.path.exists(target_path):
        return target_path
    
    return model_name

def train(args):
    # 1. Resolve model path
    model_path = get_weights_path(args.model)
    print(f"🚀 Loading model: {model_path}")

    # 2. Resolve dataset YAML path (Ultralytics uses global settings_dir, not YAML location)
    resolved_yaml = resolve_data_yaml(args.data)
    tmp_yaml = resolved_yaml if resolved_yaml != args.data else None

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model '{model_path}': {e}")
        if tmp_yaml:
            os.unlink(tmp_yaml)
        return

    # 3. Train the model
    print(f"🏋️ Starting training on {args.device}...")
    try:
        results = model.train(
            data=resolved_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            pose=True,
            patience=50,
            save=True,
            exist_ok=True,
        )
        print(f"✅ Training completed. Best model saved at: {results.save_dir}")

        # Move downloaded base weights to assets/weights if needed
        if not os.path.isabs(model_path) and os.path.exists(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "../../"))
            dest = os.path.join(project_root, "assets/weights", model_path)
            print(f"📦 Moving downloaded base weights to {dest}...")
            shutil.move(model_path, dest)
    finally:
        if tmp_yaml:
            os.unlink(tmp_yaml)

if __name__ == "__main__":
    # Default configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../../datasets/pitch_keypoints_26pt/data.yaml")

    parser = argparse.ArgumentParser(description="Train YOLO Pose Estimator")
    parser.add_argument("--data", type=str, default=config_path, help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolo26m-pose.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu, cuda, mps)")
    parser.add_argument("--project", type=str, default="runs/pose", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="pitch_pose", help="Experiment name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Config file not found at {args.data}")
        exit(1)

    train(args)
