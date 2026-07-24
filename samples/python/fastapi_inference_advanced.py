from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import logging
import time

from openvino.runtime import Core

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load OpenVINO model
core = Core()
model = core.compile_model("model.xml", "CPU")

# Input schema
class InputData(BaseModel):
    data: list


@app.post("/infer")
async def infer(input_data: InputData):
    logging.info("Received inference request")

    # Convert input
    np_input = np.array(input_data.data, dtype=np.float32)

    # Basic batching support
    if np_input.ndim == 3:
        np_input = np.expand_dims(np_input, axis=0)

    start_time = time.time()

    # Run inference
    result = model([np_input])

    inference_time = time.time() - start_time
    logging.info(f"Inference completed in {inference_time:.4f} seconds")

    return {
        "result": result[0].tolist(),
        "inference_time": inference_time
    }