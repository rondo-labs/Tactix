from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26-pose.pt")

    model.train(
        data="pitch_keypoints_26pt/data.yaml", 
        epochs=100,
        imgsz=640,
        batch=32, 
        device=0, 
        workers=8, 
        project="Tactix_Models",
        name="pose_v1"
    )

