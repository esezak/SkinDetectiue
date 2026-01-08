from ultralytics import YOLO

def train_yolo_classifier(
        data_dir="Data",
        model_name="yolo11n-cls.pt",
        epochs=50,
        imgsz=416,
        batch=16,
        device=0
):

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

    print("\nTraining complete! Check the 'runs/classify/train' folder for results.")


if __name__ == "__main__":
    # Example usage with default values
    train_yolo_classifier(
        data_dir="Data",
        model_name="yolo11n-cls.pt",
        epochs=80,
        imgsz=416,
        batch=30,
        device=0
    )
