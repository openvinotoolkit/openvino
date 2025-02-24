from __future__ import annotations
from openvino._ov_api import AsyncInferQueue
from openvino._ov_api import CompiledModel
from openvino._ov_api import Core
from openvino._ov_api import InferRequest
from openvino._ov_api import Model
from openvino._ov_api import compile_model
from openvino.utils.data_helpers.wrappers import tensor_from_file
__all__ = ['AsyncInferQueue', 'CompiledModel', 'Core', 'InferRequest', 'Model', 'compile_model', 'tensor_from_file']
