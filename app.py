
import json
import io
from pathlib import Path

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

DEVICE = torch.device("cpu")

app = FastAPI(
    title="DermaScan AI Research Prototype",
    version="2.0.0"
)

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

MODEL = None
CLASS_NAMES = []
IMAGE_SIZE = 224

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


def load_model():
    global MODEL, CLASS_NAMES

    import gc

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if not WEIGHTS.exists() or not CLASSES.exists():
        MODEL, CLASS_NAMES = None, []
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
        MODEL, CLASS_NAMES = None, []
        print(f"Model could not be loaded: {exc}")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = PREPROCESS(image).unsqueeze(0)

        with torch.inference_mode():
            outputs = MODEL(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        confidence, index = torch.max(probabilities, dim=0)

        return {
            "prediction": CLASS_NAMES[index.item()],
            "confidence": round(confidence.item() * 100, 2),
            "disclaimer": "Experimental research prototype, not a medical diagnosis."
        }

    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file.")
