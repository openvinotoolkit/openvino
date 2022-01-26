# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import copy
from typing import Any, List, Type, Union

from openvino.pyopenvino import Model
from openvino.pyopenvino import Core as CoreBase
from openvino.pyopenvino import CompiledModel as CompiledModelBase
from openvino.pyopenvino import InferRequest as InferRequestBase
from openvino.pyopenvino import AsyncInferQueue as AsyncInferQueueBase
from openvino.pyopenvino import ConstOutput
from openvino.pyopenvino import Tensor
from openvino.pyopenvino import OVAny as OVAnyBase

from openvino.runtime.utils.types import get_dtype


def tensor_from_file(path: str) -> Tensor:
    """Create Tensor from file. Data will be read with dtype of unit8."""
    return Tensor(np.fromfile(path, dtype=np.uint8))


def convert_dict_items(inputs: dict, py_types: dict) -> dict:
    """Helper function converting dictionary items to Tensors."""
    # Create new temporary dictionary.
    # new_inputs will be used to transfer data to inference calls,
    # ensuring that original inputs are not overwritten with Tensors.
    new_inputs = {}
    for k, val in inputs.items():
        if not isinstance(k, (str, int, ConstOutput)):
            raise TypeError("Incompatible key type for tensor: {}".format(k))
        try:
            ov_type = py_types[k]
        except KeyError:
            raise KeyError("Port for tensor {} was not found!".format(k))
        # Convert numpy arrays or copy Tensors
        new_inputs[k] = (
            val
            if isinstance(val, Tensor)
            else Tensor(np.array(val, get_dtype(ov_type)))
        )
    return new_inputs


def normalize_inputs(inputs: Union[dict, list], py_types: dict) -> dict:
    """Normalize a dictionary of inputs to Tensors."""
    if isinstance(inputs, dict):
        return convert_dict_items(inputs, py_types)
    elif isinstance(inputs, list):
        # Lists are required to be represented as dictionaries with int keys
        return convert_dict_items(
            {index: input for index, input in enumerate(inputs)}, py_types
        )
    else:
        raise TypeError(
            "Inputs should be either list or dict! Current type: {}".format(
                type(inputs)
            )
        )


def get_input_types(obj: Union[InferRequestBase, CompiledModelBase]) -> dict:
    """Map all tensor names of all inputs to the data types of those tensors."""

    def get_inputs(obj: Union[InferRequestBase, CompiledModelBase]) -> list:
        return obj.model_inputs if isinstance(obj, InferRequestBase) else obj.inputs

    def map_tensor_names_to_types(input: ConstOutput) -> dict:
        return {n: input.get_element_type() for n in input.get_names()}

    input_types: dict = {}
    for idx, input in enumerate(get_inputs(obj)):
        # Add all possible "accessing aliases" to dictionary
        # Key as a ConstOutput port
        input_types[input] = input.get_element_type()
        # Key as an integer
        input_types[idx] = input.get_element_type()
        # Multiple possible keys as Tensor names
        input_types.update(map_tensor_names_to_types(input))
    return input_types


class InferRequest(InferRequestBase):
    """InferRequest wrapper."""

    def infer(self, inputs: Union[dict, list] = None) -> dict:
        """Infer wrapper for InferRequest."""
        return super().infer(
            {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        )

    def start_async(
        self, inputs: Union[dict, list] = None, userdata: Any = None
    ) -> None:
        """Asynchronous infer wrapper for InferRequest."""
        super().start_async(
            {} if inputs is None else normalize_inputs(inputs, get_input_types(self)),
            userdata,
        )


class CompiledModel(CompiledModelBase):
    """CompiledModel wrapper."""

    def create_infer_request(self) -> InferRequest:
        """Create new InferRequest object."""
        return InferRequest(super().create_infer_request())

    def infer_new_request(self, inputs: Union[dict, list] = None) -> dict:
        """Infer wrapper for CompiledModel."""
        return super().infer_new_request(
            {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        )

    def __call__(self, inputs: Union[dict, list] = None) -> dict:
        """Callable infer wrapper for CompiledModel."""
        return self.infer_new_request(inputs)


class AsyncInferQueue(AsyncInferQueueBase):
    """AsyncInferQueue wrapper."""

    def __getitem__(self, i: int) -> InferRequest:
        """Return i-th InferRequest from AsyncInferQueue."""
        return InferRequest(super().__getitem__(i))

    def start_async(
        self, inputs: Union[dict, list] = None, userdata: Any = None
    ) -> None:
        """Asynchronous infer wrapper for AsyncInferQueue."""
        super().start_async(
            {}
            if inputs is None
            else normalize_inputs(
                inputs, get_input_types(self[self.get_idle_request_id()])
            ),
            userdata,
        )


class Core(CoreBase):
    """Core wrapper."""

    def compile_model(
        self, model: Union[Model, str], device_name: str = None, config: dict = None
    ) -> CompiledModel:
        """Compile a model from given Model."""
        if device_name is None:
            return CompiledModel(
                super().compile_model(model, {} if config is None else config)
            )

        return CompiledModel(
            super().compile_model(model, device_name, {} if config is None else config)
        )

    def import_model(
        self, model_file: str, device_name: str, config: dict = None
    ) -> CompiledModel:
        """Compile a model from given model file path."""
        return CompiledModel(
            super().import_model(
                model_file, device_name, {} if config is None else config
            )
        )


class ExtendedNetwork(CompiledModel):
    """CompiledModel that additionally holds Core object."""

    def __init__(self, core: Core, net: CompiledModel):
        super().__init__(net)
        self.core = core  # needs to store Core object for CPU plugin


def compile_model(model_path: str) -> CompiledModel:
    """Compact method to compile model with AUTO plugin."""
    core = Core()
    return ExtendedNetwork(core, core.compile_model(model_path, "AUTO"))


class OVAny(OVAnyBase):
    """OVAny wrapper.

    Wrapper provides some useful overloads for simple built-in Python types.

    Access to the OVAny value is direct if it is a built-in Python data type.
    Example:
    @code{.py}
        any = OVAny([1, 2])
        print(any[0])

        Output: 2
    @endcode

    Otherwise if OVAny value is a custom data type (for example user class),
    access to the value is possible by 'get()' method or property 'value'.
    Example:
    @code{.py}
        class Test:
            def __init__(self):
                self.data = "test"

        any = OVAny(Test())
        print(any.value.data)
    @endcode

    """

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self.value[key]

    def __get__(self) -> Any:
        return self.value

    def __setitem__(self, key: Union[str, int], val: Any) -> None:
        self.value[key] = val

    def __set__(self, val: Any) -> None:
        self.value = val

    def __len__(self) -> int:
        return len(self.value)
