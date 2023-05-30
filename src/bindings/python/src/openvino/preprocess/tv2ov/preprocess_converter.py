from typing import Callable, Any
import logging

import openvino.runtime as ov


class PreprocessConvertor():
    def __init__(self, model: ov.Model):
        self._model = model

    @staticmethod
    def from_torchvision(model: ov.Model, transform: Callable, input_example: Any,
                         input_name: str = None) -> ov.Model:
        """
        Convert torchvision transform to OpenVINO preprocessing
        Arguments:
            model (ov.Model):
                Result name
            transform (Callable):
                torchvision transform to convert
            input_example (torch.Tensor or np.ndarray or PIL.Image):
                Example of input data for transform to trace its structure.
                Don't confuse with the model input.
            input_name (str):
                Name of the result node to connect with preprocessing.
                It can be None if the model has one input.
        Returns:
            ov.Mode: OpenVINO Model object with preprocessing
        Example:
            >>> model = PreprocessorConvertor.from_torchvision(model, "input", transform, input_example)
        """
        try:
            from torchvision import transforms
            from torchvision_preprocessing import from_torchvision
            return from_torchvision(model, transform, input_example, input_name)
        except ImportError:
            raise ImportError("Please install torchvision")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise e
