from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from openvino.runtime import Core
import numpy as np
import cv2

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "public"
    / "mobilenet-v2-pytorch"
    / "FP16"
    / "mobilenet-v2-pytorch.xml"
)

# ---------------- Globals ----------------
IMAGENET_LABELS = []
compiled_model = None
input_layer = None
output_layer = None


# ---------------- Lifespan (startup replacement) ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_model, input_layer, output_layer, IMAGENET_LABELS

    # Load ImageNet labels
    labels_path = BASE_DIR / "imagenet_labels.txt"
    if not labels_path.exists():
        raise RuntimeError("imagenet_labels.txt not found")

    IMAGENET_LABELS = labels_path.read_text().splitlines()
    print(f"Loaded {len(IMAGENET_LABELS)} ImageNet labels ✅")

    # Load and compile model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    core = Core()
    model = core.read_model(MODEL_PATH)
    compiled_model = core.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print("Model loaded and compiled ✅")
    yield


# ---------------- FastAPI app ----------------
app = FastAPI(
    title="OpenVINO FastAPI Inference",
    lifespan=lifespan
)


# ---------------- Root endpoint ----------------
@app.get("/")
def root():
    return {"status": "FastAPI + OpenVINO is running ✅"}


# ---------------- Image preprocessing ----------------
def preprocess_image(image_bytes: bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    # Resize
    img = cv2.resize(img, (224, 224))

    # Convert to float32
    img = img.astype(np.float32)

    # Normalize (Open Model Zoo)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    scale = np.array([58.624, 57.12, 57.375], dtype=np.float32)
    img = (img - mean) / scale

    # HWC -> CHW
    img = img.transpose(2, 0, 1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# ---------------- Image inference endpoint ----------------
@app.post("/infer-image")
async def infer_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid image file: {e}"
        )

    result = compiled_model([input_tensor])
    output = result[output_layer][0]

    top5 = np.argsort(output)[-5:][::-1]

    predictions = [
        {
            "class_id": int(idx),
            "label": IMAGENET_LABELS[idx],
            "score": float(output[idx]),
        }
        for idx in top5
    ]

    return {"top5_predictions": predictions}