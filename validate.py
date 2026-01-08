from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ultralytics import YOLO

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    plt = None  # type: ignore
    _HAS_MPL = False


IMAGE_EXTS_DEFAULT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ImagePrediction:
    path: str
    true_class: str
    pred_class: str
    confidence: float


def _iter_images(class_dir: Path, exts: Sequence[str]) -> Iterable[Path]:
    # Non-recursive (matches common classifier dataset layout)
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def find_latest_best_pt(runs_dir: Path = Path("runs") / "classify") -> Optional[Path]:
    """Find the newest runs/classify/train*/weights/best.pt by modified time."""
    if not runs_dir.exists():
        return None

    candidates = list(runs_dir.glob("train*/weights/best.pt"))
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_confusion_matrix_png(
    cm: List[List[int]],
    class_names: Sequence[str],
    out_path: Path,
) -> bool:
    """Save a confusion matrix visualization as a PNG.
    Returns True if saved, False if matplotlib isn't available.
    """
    if not _HAS_MPL or plt is None:
        return False

    n = len(class_names)
    fig_w = max(8.0, min(0.5 * n + 4.0, 40.0))
    fig_h = max(6.0, min(0.5 * n + 3.0, 40.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set(
        xticks=list(range(n)),
        yticks=list(range(n)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells if matrix is not too large
    if n <= 40:
        thresh = max(1, max((max(r) if r else 0) for r in cm) // 2)
        for i in range(n):
            for j in range(n):
                v = cm[i][j]
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v > thresh else "black",
                )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def validate_dataset(
    model_path: Optional[str] = None,
    test_dir: str = "Data/test",
    device: int = 0,
    exts: Sequence[str] = IMAGE_EXTS_DEFAULT,
    save_report: bool = True,
    report_root: str = "runs/validate",
) -> int:
    """Custom validation loop over every image in test_dir.
    Returns process exit code: 0 if ran successfully, 2 if structural issues (missing model/dir), 3 if no images.
    """

    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"ERROR: Test directory not found: {test_dir}", file=sys.stderr)
        return 2

    resolved_model: Optional[Path]
    if model_path:
        resolved_model = Path(model_path)
    else:
        resolved_model = find_latest_best_pt()

    if not resolved_model or not resolved_model.exists():
        print(
            "ERROR: Model not found. Provide --model or ensure runs/classify/train*/weights/best.pt exists.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading model from: {resolved_model}")
    model = YOLO(str(resolved_model))

    # Collect classes from test dataset folder names
    class_dirs = [p for p in sorted(test_path.iterdir()) if p.is_dir()]
    if not class_dirs:
        print(f"WARN: No class folders found under: {test_path}")

    known_classes = [p.name for p in class_dirs]
    known_set = set(known_classes)

    # Confusion matrix: rows=true, cols=pred, only for known classes
    conf: Dict[str, Dict[str, int]] = {t: {p: 0 for p in known_classes} for t in known_classes}
    per_class_totals: Dict[str, int] = {c: 0 for c in known_classes}
    per_class_correct: Dict[str, int] = {c: 0 for c in known_classes}

    predictions: List[ImagePrediction] = []

    total = 0
    correct = 0

    for class_dir in class_dirs:
        true_class = class_dir.name
        images = list(_iter_images(class_dir, exts))
        if not images:
            print(f"WARN: No images found in class: {true_class} (skipping)")
            continue

        for img_path in images:
            try:
                result = model.predict(source=str(img_path), device=device, verbose=False)[0]
                pred_class = result.names[result.probs.top1]
                conf_score = float(result.probs.top1conf.item())
            except Exception as e:
                print(f"WARN: Failed to predict {img_path}: {e} (skipping)")
                continue

            total += 1
            per_class_totals[true_class] = per_class_totals.get(true_class, 0) + 1

            if pred_class not in known_set:
                print(
                    f"WARN: Predicted class '{pred_class}' not found in test folders for image {img_path.name} "
                    f"(true '{true_class}'). Skipping from metrics.")
                # Still record prediction for debugging
                predictions.append(
                    ImagePrediction(
                        path=str(img_path),
                        true_class=true_class,
                        pred_class=pred_class,
                        confidence=conf_score,
                    )
                )
                continue

            conf[true_class][pred_class] += 1

            is_correct = pred_class == true_class
            if is_correct:
                correct += 1
                per_class_correct[true_class] = per_class_correct.get(true_class, 0) + 1

            predictions.append(
                ImagePrediction(
                    path=str(img_path),
                    true_class=true_class,
                    pred_class=pred_class,
                    confidence=conf_score,
                )
            )

    if total == 0:
        print("ERROR: No images were evaluated.", file=sys.stderr)
        return 3

    acc = correct / total
    print("\n=== Validation Summary ===")
    print(f"Test dir:   {test_path}")
    print(f"Model:      {resolved_model}")
    print(f"Device:     {device}")
    print(f"Evaluated:  {total} images")
    print(f"Correct:    {correct}")
    print(f"Accuracy:   {acc:.4f}")

    print("\nPer-class accuracy (evaluated images only):")
    for c in known_classes:
        n = per_class_totals.get(c, 0)
        if n == 0:
            print(f"  {c}: n=0 (skipped)")
            continue
        pc = per_class_correct.get(c, 0)
        print(f"  {c}: {pc}/{n} = {pc / n:.4f}")

    # Save report
    if save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(report_root) / timestamp
        _safe_mkdir(out_dir)

        report = {
            "timestamp": timestamp,
            "test_dir": str(test_path),
            "model": str(resolved_model),
            "device": device,
            "evaluated_images": total,
            "correct": correct,
            "accuracy": acc,
            "classes": known_classes,
            "per_class_totals": per_class_totals,
            "per_class_correct": per_class_correct,
            "confusion_matrix": conf,
        }

        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Per-image predictions CSV
        with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "true_class", "pred_class", "confidence"])
            for p in predictions:
                w.writerow([p.path, p.true_class, p.pred_class, f"{p.confidence:.6f}"])

        # Confusion matrix PNG
        cm = [[conf[t][p] for p in known_classes] for t in known_classes]
        cm_path = out_dir / "confusion_matrix.png"
        if _save_confusion_matrix_png(cm, known_classes, cm_path):
            print(f"Saved confusion matrix PNG to: {cm_path}")
        else:
            print(
                "WARN: matplotlib not available; skipping confusion_matrix.png generation. "
                "Install it with: pip install matplotlib"
            )

        print(f"\nSaved report to: {out_dir}")

    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate a trained YOLO classifier on Data/test.")
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained weights (.pt). If omitted, picks newest runs/classify/train*/weights/best.pt",
    )
    ap.add_argument("--test_dir", type=str, default="Data/test", help="Test dataset directory.")
    ap.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device (0 for GPU, -1 for CPU). Default matches predictScript.py (0).",
    )
    ap.add_argument(
        "--no_report",
        action="store_true",
        help="Disable saving JSON/CSV report under runs/validate/<timestamp>/.",
    )
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(
        validate_dataset(
            model_path=args.model,
            test_dir=args.test_dir,
            device=args.device,
            save_report=(not args.no_report),
        )
    )
