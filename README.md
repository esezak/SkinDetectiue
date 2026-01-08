# SkinDetect – Model Validation

This repo trains a YOLO **image classification** model (Ultralytics) and includes a custom validation script that evaluates the trained model on the **entire** test dataset.

## What `validate.py` does

`validate.py` runs a **custom loop** over every image in:

```
Data/test/<class_name>/*
```

For each image it:
- loads the trained model (`.pt`)
- runs inference
- compares the predicted class name to the ground-truth class (the folder name)

It prints:
- overall accuracy
- per-class accuracy

It also optionally saves a report:
- `runs/validate/<timestamp>/report.json`
- `runs/validate/<timestamp>/predictions.csv`
- `runs/validate/<timestamp>/confusion_matrix.png` *(requires matplotlib)*

### Confusion matrix PNG

If `matplotlib` is installed, `validate.py` will render a confusion matrix image:

- `runs/validate/<timestamp>/confusion_matrix.png`

If `matplotlib` is not available, the script will print a warning and skip PNG generation.

Install it with:

```powershell
pip install matplotlib
```

### Default model selection

If you do **not** pass `--model`, it automatically selects the **newest**:

```
runs/classify/train*/weights/best.pt
```

(newest = most recently modified).

### “Warn and skip” behavior

The script will **warn and skip** instead of crashing when:
- a class folder contains no images
- an image fails to load / predict
- the model predicts a class name that doesn’t exist as a folder under `Data/test`

## Requirements

- Python environment with `ultralytics` installed.
- A trained model checkpoint (`best.pt`) produced by training.
- *(Optional)* `matplotlib` for `confusion_matrix.png` output.

If you trained using `trainScript.py`, Ultralytics typically writes checkpoints under:

```
runs/classify/train*/weights/best.pt
```

## Run validation

From the project root (same folder as `validate.py`):

### GPU (default)

```powershell
python .\validate.py
```

### CPU

```powershell
python .\validate.py --device -1
```

### Use a specific model file

```powershell
python .\validate.py --model runs\classify\train7\weights\best.pt
```

### Disable report outputs

```powershell
python .\validate.py --no_report
```

## Output files

When reporting is enabled (default), a new folder is created:

```
runs/validate/<timestamp>/
```

- `report.json`: summary results (accuracy, per-class totals, confusion matrix)
- `predictions.csv`: per-image results (path, true_class, pred_class, confidence)
- `confusion_matrix.png`: confusion matrix visualization *(requires matplotlib)*

## Notes / troubleshooting

- **CUDA vs CPU:** Use `--device 0` for the first GPU, or `--device -1` for CPU.
- **Auto-install warnings:** Ultralytics may auto-install small dependencies the first time you run; re-run the command if it asks you to restart.
- **Dataset layout matters:** `validate.py` assumes a standard classification folder layout: each class is a folder under `Data/test`.

## Related scripts

- `trainScript.py`: trains a classification model
- `predictScript.py`: predicts a few random samples per class
- `validate.py`: evaluates *all* test images and reports accuracy

