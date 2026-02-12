import os
from fastapi import FastAPI, UploadFile, File
from openvino.runtime import Core
import numpy as np
import cv2

# ---------------- FastAPI app ----------------
app = FastAPI(title="OpenVINO FastAPI Inference")

# ---------------- Global variables ----------------
IMAGENET_LABELS = []
core = Core()
compiled_model = None
input_layer = None
output_layer = None

# ---------------- Startup event: load labels & model ----------------
@app.on_event("startup")
def startup_event():
    global IMAGENET_LABELS, compiled_model, input_layer, output_layer

    # Load ImageNet labels
    base_dir = os.path.dirname(os.path.abspath(__file__))
    labels_path = os.path.join(base_dir, "imagenet_labels.txt")
    with open(labels_path, "r") as f:
        IMAGENET_LABELS = [line.strip() for line in f]
    print(f"Loaded {len(IMAGENET_LABELS)} ImageNet labels ✅")

    # Load and compile OpenVINO model
    model_path = "public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml"
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    print(f"Model loaded and compiled ✅")


# ---------------- Root endpoint ----------------
@app.get("/")
def root():
    return {"status": "FastAPI + OpenVINO is running ✅"}


# ---------------- Image preprocessing ----------------
def preprocess_image(image_bytes: bytes):
    # Decode image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    # Resize to model input size
    img = cv2.resize(img, (224, 224))

    # Convert to float32
    img = img.astype(np.float32)

    # Normalize (Open Model Zoo standard)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    scale = np.array([58.624, 57.12, 57.375], dtype=np.float32)
    img = (img - mean) / scale

    # HWC → CHW
    img = img.transpose(2, 0, 1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# ---------------- Image inference endpoint ----------------
@app.post("/infer-image")
async def infer_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    result = compiled_model([input_tensor])
    output = result[output_layer][0]

    top5 = np.argsort(output)[-5:][::-1]

    predictions = []
    for idx in top5:
        predictions.append({
            "class_id": int(idx),
            "label": IMAGENET_LABELS[idx],
            "score": float(output[idx])
        })

    return {
        "top5_predictions": predictions
    }
