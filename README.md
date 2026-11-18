# DermaScan AI v2 — Research Prototype

A FastAPI + PyTorch starter project for educational skin-image classification research.

## Important limitations
- This is **not a medical device** and does not diagnose, treat, or rule out disease.
- No model weights or training dataset are included. The application starts safely in `demo_untrained` mode.
- A CNN cannot be made 100% accurate. Never claim 100% accuracy.
- The inference score is a model score, not the probability that a person has a disease.
- The simple score threshold in `app.py` is only a demo guard. It is not validated out-of-distribution detection or clinical calibration.

## Run locally (Windows PowerShell)
Use Python 3.11 or 3.12 (64-bit). Avoid preview Python versions.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000 and http://127.0.0.1:8000/health.

## Dataset layout
Put images into folders named exactly as the category labels:

```text
training/data/
  class_a/
    image001.jpg
  class_b/
    image002.jpg
```

Use only images you are permitted to use. Confirm diagnoses/labels and licensing. Keep all images from the same patient in only one split; otherwise data leakage can inflate results. Do not use images from the independent test set during training or model selection.

## Train
```powershell
python training/train.py --data training/data --epochs 10 --batch-size 16
```
The script makes a stratified random image-level split for a basic demonstration. For real research, replace it with a patient-level split using patient IDs and an independent external test cohort. Do not treat the default split as clinical validation.

Training writes `models/dermascan_resnet18.pth` and `models/class_names.json`.

## Evaluate
```powershell
python training/evaluate.py --data training/test_data
```
`training/test_data` must contain a separate, untouched test folder with the same class-folder structure. Metrics are reported per class and overall. Only report metrics from a suitable independent test set.

## Deploy to Render
Push the project to GitHub, create a Render Web Service from the repository, and use the included `render.yaml`. Large model weights may exceed free deployment limits; use an appropriate model-artifact store or paid persistent storage if needed. Never commit private patient data or identifiable images.

## Before any real-world use
Obtain ethics/privacy approvals as applicable, verify dataset provenance and consent, use patient-level and external validation, test across skin tones/cameras/clinical settings, calibrate confidence, assess subgroup performance, and obtain qualified clinical review.

## Quick demo training (CPU-friendly)
The included quick trainer defaults to sampling up to **10 images per class**, training only the final ResNet18 classification layer, and running **1 epoch**. This is a software demo checkpoint—not a medically meaningful classifier.

From the project folder in PowerShell:

```powershell
python training/train.py --data training/data --epochs 1 --batch-size 8 --max-per-class 10
```

It saves `models/dermascan_resnet18.pth` and `models/class_names.json`. The dataset is not bundled with this ZIP; keep your existing `training/data` folder and its labeled class folders when replacing project files. If the pretrained-weight download is already cached locally, it should not download again.

Run the web app:

```powershell
uvicorn app:app --reload
```
Then open http://127.0.0.1:8000. The model status will indicate whether the checkpoint was loaded successfully. Results are experimental only and must not be used for health decisions.
