import os
import random
from ultralytics import YOLO
from pathlib import Path

def predict_random_samples(
        model_path="runs/classify/train3/weights/best.pt",
        test_dir="database/test",
        samples_per_class=5,
        device=0  # set -1 for CPU
):
    """
    Selects N random images per class from test_dir and predicts their class using a trained YOLO classifier.

    Args:
        model_path (str): Path to the trained YOLO classifier model (.pt file).
        test_dir (str): Path to the test dataset folder (each class has its own subfolder).
        samples_per_class (int): Number of random samples per class.
        device (int): CUDA device (0 for GPU, -1 for CPU).
    """

    # Load trained model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    for class_folder in test_path.iterdir():
        if not class_folder.is_dir():
            continue

        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpeg"))
        if len(images) == 0:
            print(f"No images found in class: {class_folder.name}")
            continue

        sample_images = random.sample(images, min(samples_per_class, len(images)))

        print(f"\nClass: {class_folder.name} | Selected {len(sample_images)} samples")

        # Run predictions
        for img_path in sample_images:
            result = model.predict(source=str(img_path), device=device, verbose=False)[0]
            predicted_class = result.names[result.probs.top1]
            confidence = result.probs.top1conf.item()
            print(f"  {img_path.name}: {predicted_class} ({confidence:.2f})")


if __name__ == "__main__":
    predict_random_samples(
        test_dir="no_augment_v1/test",
        samples_per_class=10,
        device=0  # change to -1 for CPU
    )
