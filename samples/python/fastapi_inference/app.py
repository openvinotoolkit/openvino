from fastapi import FastAPI, HTTPException
import numpy as np
import os

from schemas import InferenceRequest, InferenceResponse
from model_utils import OpenVINOModel

MODEL_PATH = os.getenv(
    "OV_MODEL_PATH",
    "models/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml"
)
DEVICE = os.getenv("OV_DEVICE", "CPU")

app = FastAPI(
    title="OpenVINO FastAPI Inference",
    description="Lightweight REST inference service using OpenVINO",
    version="1.0.0"
)

ov_model = None


@app.on_event("startup")
def load_model():
    global ov_model
    try:
        ov_model = OpenVINOModel(MODEL_PATH, DEVICE)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Model not loaded: {e}")


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    if ov_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        inputs = np.array(request.inputs, dtype=np.float32)
        outputs = ov_model.infer(inputs)
        return InferenceResponse(outputs=outputs.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))