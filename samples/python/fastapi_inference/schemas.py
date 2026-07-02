from pydantic import BaseModel
from typing import List

class InferenceRequest(BaseModel):
    inputs: List[List[List[List[float]]]]

class InferenceResponse(BaseModel):
    outputs: List[List[float]]