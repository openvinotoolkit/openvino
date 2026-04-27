from openvino.runtime import Core
import numpy as np

class OpenVINOModel:
    def __init__(self, model_path: str, device: str = "CPU"):
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def infer(self, inputs: np.ndarray):
        if inputs.ndim != 4:
            raise ValueError("Input must be a 4D tensor [N, C, H, W]")

        result = self.compiled_model([inputs])
        return result[self.output_layer]