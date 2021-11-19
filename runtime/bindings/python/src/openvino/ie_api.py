# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import copy
from typing import Any, List, Union

from openvino.pyopenvino import Function
from openvino.pyopenvino import Core as CoreBase
from openvino.pyopenvino import ExecutableNetwork as ExecutableNetworkBase
from openvino.pyopenvino import InferRequest as InferRequestBase
from openvino.pyopenvino import AsyncInferQueue as AsyncInferQueueBase
from openvino.pyopenvino import Tensor

from openvino.utils.types import get_dtype


def tensor_from_file(path: str) -> Tensor:
    """Create Tensor from file. Data will be read with dtype of unit8."""
    return Tensor(np.fromfile(path, dtype=np.uint8))


def normalize_inputs(py_dict: dict, py_types: dict) -> dict:
    """Normalize a dictionary of inputs to Tensors."""
    for k, val in py_dict.items():
        try:
            if isinstance(k, int):
                ov_type = list(py_types.values())[k]
            elif isinstance(k, str):
                ov_type = py_types[k]
            else:
                raise TypeError("Incompatible key type for tensor named: {}".format(k))
        except KeyError:
            raise KeyError("Port for tensor named {} was not found!".format(k))
        py_dict[k] = val if isinstance(val, Tensor) else Tensor(np.array(val, get_dtype(ov_type)))
    return py_dict


def get_input_types(obj: Union[InferRequestBase, ExecutableNetworkBase]) -> dict:
    """Get all precisions from object inputs."""
    return {i.get_node().get_friendly_name(): i.get_node().get_element_type() for i in obj.inputs}


class InferRequest(InferRequestBase):
    """InferRequest wrapper."""

    def infer(self, inputs: dict = None) -> List[np.ndarray]:
        """Infer wrapper for InferRequest."""
        inputs = {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        res = super().infer(inputs)
        # Required to return list since np.ndarray forces all of tensors data to match in
        # dimensions. This results in errors when running ops like variadic split.
        return [copy.deepcopy(tensor.data) for tensor in res]

    def start_async(self, inputs: dict = None, userdata: Any = None) -> None:
        """Asynchronous infer wrapper for InferRequest."""
        inputs = {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        super().start_async(inputs, userdata)


class ExecutableNetwork(ExecutableNetworkBase):
    """ExecutableNetwork wrapper."""

    def create_infer_request(self) -> InferRequest:
        """Create new InferRequest object."""
        return InferRequest(super().create_infer_request())

    def infer_new_request(self, inputs: dict = None) -> List[np.ndarray]:
        """Infer wrapper for ExecutableNetwork."""
        inputs = {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        res = super().infer_new_request(inputs)
        # Required to return list since np.ndarray forces all of tensors data to match in
        # dimensions. This results in errors when running ops like variadic split.
        return [copy.deepcopy(tensor.data) for tensor in res]


class AsyncInferQueue(AsyncInferQueueBase):
    """AsyncInferQueue wrapper."""

    def __getitem__(self, i: int) -> InferRequest:
        """Return i-th InferRequest from AsyncInferQueue."""
        return InferRequest(super().__getitem__(i))

    def start_async(self, inputs: dict = None, userdata: Any = None) -> None:
        """Asynchronous infer wrapper for AsyncInferQueue."""
        inputs = (
            {}
            if inputs is None
            else normalize_inputs(inputs, get_input_types(self[self.get_idle_request_id()]))
        )
        super().start_async(inputs, userdata)


class Core(CoreBase):
    """Core wrapper."""

    def compile_model(
        self, model: Union[Function, str], device_name: str, config: dict = None
    ) -> ExecutableNetwork:
        """Compile a model from given Function."""
        return ExecutableNetwork(
            super().compile_model(model, device_name, {} if config is None else config)
        )

    def import_model(
        self, model_file: str, device_name: str, config: dict = None
    ) -> ExecutableNetwork:
        """Compile a model from given model file path."""
        return ExecutableNetwork(
            super().import_model(model_file, device_name, {} if config is None else config)
        )


class ExtendedNetwork(ExecutableNetwork):
    """ExecutableNetwork that additionally holds Core object."""

    def __init__(self, core: Core, net: ExecutableNetwork):
        super().__init__(net)
        self.core = core  # needs to store Core object for CPU plugin


def compile_model(model_path: str) -> ExecutableNetwork:
    """Compact method to compile model with AUTO plugin."""
    core = Core()
    return ExtendedNetwork(core, core.compile_model(model_path, "AUTO"))
