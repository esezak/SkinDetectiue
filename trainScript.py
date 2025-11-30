from ultralytics import YOLO
import os


def train_yolo_classifier(
        data_dir="datasets/garbage_classification",
        model_name="yolo11n-cls.pt",  #yolo11s-cls.pt, yolo11m-cls.pt
        epochs=50,
        imgsz=224,
        batch=16,
        device=0
):

    # Check dataset existence
    if not os.path.exists(os.path.join(data_dir, "train")) or not os.path.exists(os.path.join(data_dir, "val")):
        raise FileNotFoundError(f"Dataset not found or not structured properly at: {data_dir}")

    # Load YOLO model
    model = YOLO(model_name)

    # Train model
    model.train(
        data=data_dir,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device
    )

    print("\n✅ Training complete! Check the 'runs/classify/train' folder for results.")


if __name__ == "__main__":
    # Example usage
    train_yolo_classifier(
        data_dir="no_augment_v1",
        model_name="yolo11n-cls.pt",
        epochs=100,
        imgsz=400,
        batch=30,
        device=0
    )
