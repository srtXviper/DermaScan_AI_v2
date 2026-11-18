
import json
import io
import gc
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import torch
from torchvision import models, transforms


# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

WEIGHTS = MODEL_DIR / "dermascan_resnet18.pth"
CLASSES = MODEL_DIR / "class_names.json"

DEVICE = torch.device("cpu")

IMAGE_SIZE = 224

MODEL = None
CLASS_NAMES = []


# ---------------- FASTAPI APP ----------------

app = FastAPI(
    title="DermaScan AI Research Prototype",
    version="2.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static"
)


# ---------------- IMAGE PREPROCESSING ----------------

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


# ---------------- MODEL LOADING ----------------

def load_model():
    global MODEL, CLASS_NAMES

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if not WEIGHTS.exists() or not CLASSES.exists():
        MODEL = None
        CLASS_NAMES = []
        print("Model weights or class names not found.")
        return

    try:
        CLASS_NAMES = json.loads(
            CLASSES.read_text(encoding="utf-8")
        )

        if not isinstance(CLASS_NAMES, list) or len(CLASS_NAMES) < 2:
            raise ValueError("Invalid class_names.json")

        model = models.resnet18(weights=None)

        model.fc = torch.nn.Linear(
            model.fc.in_features,
            len(CLASS_NAMES)
        )

        state = torch.load(
            WEIGHTS,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(state)

        del state
        gc.collect()

        model.eval()
        MODEL = model

        print("DermaScan model loaded successfully.")

    except Exception as exc:
        MODEL = None
        CLASS_NAMES = []
        print(f"Model could not be loaded: {exc}")


# ---------------- STARTUP ----------------

@app.on_event("startup")
def startup_event():
    load_model()


# ---------------- HOME PAGE ----------------

@app.get("/")
def home():
    return FileResponse(
        ROOT / "static" / "index.html"
    )


# ---------------- HEALTH CHECK ----------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": MODEL is not None,
        "classes": len(CLASS_NAMES)
    }


# ---------------- IMAGE PREDICTION ----------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available."
        )

    try:
        contents = await file.read()

        if len(contents) > 8 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 8 MB limit."
            )

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        tensor = PREPROCESS(image).unsqueeze(0)

        with torch.inference_mode():
            outputs = MODEL(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top_count = min(3, len(CLASS_NAMES))

        scores, indices = torch.topk(
            probabilities,
            top_count
        )

        top_scores = []

        for score, index in zip(scores, indices):
            top_scores.append({
                "category": CLASS_NAMES[index.item()],
                "score": round(score.item(), 6)
            })

        return {
            "status": "prediction_available",
            "prediction": {
                "category": CLASS_NAMES[indices[0].item()],
                "model_score": round(scores[0].item(), 6),
                "top_scores": top_scores
            },
            "warning": (
                "Experimental research output only. "
                "Not a medical diagnosis. Consult a qualified "
                "healthcare professional for medical concerns."
            )
        }

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    finally:
        if "contents" in locals():
            del contents


# ---------------- SERVER STATUS ----------------

@app.get("/favicon.ico")
def favicon():
    raise HTTPException(status_code=404)
