import json, io, os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
WEIGHTS = MODEL_DIR / "dermascan_resnet18.pth"
CLASSES = MODEL_DIR / "class_names.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="DermaScan AI Research Prototype", version="2.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

MODEL = None
CLASS_NAMES = []
IMAGE_SIZE = 224
PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model():
    global MODEL, CLASS_NAMES
    if not WEIGHTS.exists() or not CLASSES.exists():
        MODEL, CLASS_NAMES = None, []
        return
    try:
        CLASS_NAMES = json.loads(CLASSES.read_text(encoding="utf-8"))
        if not isinstance(CLASS_NAMES, list) or len(CLASS_NAMES) < 2:
            raise ValueError("class_names.json must contain at least two class names")
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        state = torch.load(WEIGHTS, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        MODEL = model
    except Exception as exc:
        MODEL, CLASS_NAMES = None, []
        print(f"Model could not be loaded: {exc}")

@app.on_event("startup")
def startup():
    load_model()

@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL is not None,
            "classes": len(CLASS_NAMES), "device": str(DEVICE),
            "mode": "trained_model" if MODEL is not None else "demo_untrained"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    raw = await file.read()
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(413, "Image must be 8 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The uploaded file is not a valid image.")
    w, h = image.size
    if min(w, h) < 96:
        raise HTTPException(400, "Image is too small. Please use an image at least 96 x 96 pixels.")
    if MODEL is None:
        return {
            "status": "model_not_trained",
            "message": "The website is running, but no verified trained model is installed yet.",
            "image": {"width": w, "height": h},
            "prediction": None,
            "next_step": "Add a properly labelled dataset, train the model, evaluate it on an independent test set, then install the checkpoint."
        }
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = MODEL(tensor)[0]
        probs = torch.softmax(logits, dim=0)
        score, idx = torch.max(probs, dim=0)
    # This is only a conservative demo gate, NOT a clinically validated rejector.
    threshold = float(os.getenv("DERMASCAN_MIN_SCORE", "0.70"))
    confidence = float(score.item())
    if confidence < threshold:
        return {"status": "uncertain", "message": "The model is not confident enough to return a category.",
                "image": {"width": w, "height": h}, "prediction": None,
                "raw_model_score": round(confidence, 4),
                "warning": "Model score is not the probability that a person has a disease."}
    return {"status": "experimental_prediction", "message": "Experimental model category only — not a diagnosis.",
            "image": {"width": w, "height": h},
            "prediction": {"category": CLASS_NAMES[int(idx)], "model_score": round(confidence, 4)},
            "warning": "Do not use this result to diagnose, treat, or rule out a skin condition."}

@app.get("/api/model-info")
def model_info():
    return {"model_ready": MODEL is not None, "architecture": "ResNet18",
            "classes": CLASS_NAMES, "device": str(DEVICE),
            "clinical_validation": False}
