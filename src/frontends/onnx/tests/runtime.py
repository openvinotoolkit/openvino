# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Provide a layer of abstraction for an OpenVINO runtime environment."""

import logging
from typing import Dict, List, Union

import numpy as np

from openvino.runtime import Core

from openvino.runtime.exceptions import UserInputError
from openvino.runtime import Model, Node, Tensor, Type
from openvino.runtime.utils.types import NumericData, get_shape, get_dtype

from onnx.helper import float32_to_float8e5m2, float32_to_float8e4m3
from onnx.numpy_helper import float8e5m2_to_float32, float8e4m3_to_float32

import tests

log = logging.getLogger(__name__)


def runtime(backend_name: str = "CPU") -> "Runtime":
    """Create a Runtime object (helper factory)."""
    return Runtime(backend_name)


def get_runtime():
    """Return runtime object."""
    if tests.BACKEND_NAME is not None:
        return runtime(backend_name=tests.BACKEND_NAME)
    else:
        return runtime()

def get_onnx_test_dtype(ov_type):
    is_ov_float8 = ov_type == Type.f8e4m3 or ov_type == Type.f8e5m2
    if is_ov_float8:
        return np.float32
    else:
        return get_dtype(ov_type)

class Runtime(object):
    """Represents a graph runtime environment."""

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name
        log.debug(f"Creating runtime for {backend_name}")
        self.backend = Core()
        assert backend_name in self.backend.available_devices, 'The requested device "' + backend_name + '" is not supported!'

    def set_config(self, config: Dict[str, str]) -> None:
        """Set the runtime configuration."""
        self.backend.set_property(device_name=self.backend_name, properties=config)

    def computation(self, node_or_model: Union[Node, Model], *inputs: Node) -> "Computation":
        """Return a callable Computation object."""
        if isinstance(node_or_model, Node):
            model = Model(node_or_model, inputs, node_or_model.name)
            return Computation(self, model)
        elif isinstance(node_or_model, Model):
            return Computation(self, node_or_model)
        else:
            raise TypeError(
                "Runtime.computation must be called with an OpenVINO Model object "
                "or an OpenVINO node object an optionally Parameter node objects. "
                "Called with: %s",
                node_or_model,
            )

    def __repr__(self) -> str:
        return f"<Runtime: Backend='{self.backend_name}'>"


class Computation(object):
    """Graph callable computation object."""

    def __init__(self, runtime: Runtime, model: Model) -> None:
        self.runtime = runtime
        self.model = model
        self.parameters = model.get_parameters()
        self.results = model.get_results()
        self.network_cache = {}

    def convert_buffers(self, source_buffers, target_dtypes):
        converted_buffers = []
        for i in range(len(source_buffers)):
            key = list(source_buffers)[i]
            target_dtype = target_dtypes[i]
            # custom conversion for bf16
            if self.results[i].get_output_element_type(0) == Type.bf16:
                converted_buffers.append((source_buffers[key].view(target_dtype)).astype(target_dtype))
            elif self.results[i].get_output_element_type(0) == Type.f8e5m2:
                data_f8 = source_buffers[key].tobytes()
                converted_buffers.append(float8e5m2_to_float32(np.frombuffer(data_f8, dtype=np.uint8), fn=False, uz=False).reshape(source_buffers[key].shape).view(target_dtype))
            elif self.results[i].get_output_element_type(0) == Type.f8e4m3:
                data_f8 = source_buffers[key].tobytes()
                converted_buffers.append(float8e4m3_to_float32(np.frombuffer(data_f8, dtype=np.uint8), fn=True, uz=False).reshape(source_buffers[key].shape).view(target_dtype))
            else:
                converted_buffers.append(source_buffers[key].astype(target_dtype))
        return converted_buffers

    def convert_to_tensors(self, input_values):
        input_tensors = []
        for parameter, input_val in zip(self.parameters, input_values):
            if not isinstance(input_val, (np.ndarray)):
                input_val = np.ndarray([], type(input_val), np.array(input_val))
            if parameter.get_output_element_type(0) == Type.bf16:
                input_tensors.append(Tensor(Type.bf16, input_val.shape))
                input_tensors[-1].data[:] = input_val.view(np.float16)
            elif parameter.get_output_element_type(0) == Type.f8e5m2:
                input_tensors.append(Tensor(Type.f8e5m2, input_val.shape))
                _float32_to_float8e5m2 = np.vectorize(float32_to_float8e5m2, excluded=["fn", "uz"])
                input_tensors[-1].data[:] = _float32_to_float8e5m2(input_val.astype(dtype=np.float32), fn=False, uz=False)
            elif parameter.get_output_element_type(0) == Type.f8e4m3:
                input_tensors.append(Tensor(Type.f8e4m3, input_val.shape))
                _float32_to_float8e4m3 = np.vectorize(float32_to_float8e4m3, excluded=["fn", "uz"])
                input_tensors[-1].data[:] = _float32_to_float8e4m3(input_val.astype(dtype=np.float32), fn=True, uz=False)
            else:
                input_tensors.append(Tensor(input_val))
        return input_tensors

    def __repr__(self) -> str:
        params_string = ", ".join([param.name for param in self.parameters])
        return f"<Computation: {self.model.get_name()}({params_string})>"

    def __call__(self, *input_values: NumericData) -> List[NumericData]:
        """Run computation on input values and return result."""
        # Input validation
        if len(input_values) < len(self.parameters):
            raise UserInputError(
                "Expected %s params, received not enough %s values.",
                len(self.parameters),
                len(input_values),
            )

        param_names = [param.friendly_name for param in self.parameters]
        input_shapes = [get_shape(input_value) for input_value in input_values]
        if self.network_cache.get(str(input_shapes)) is None:
            model = self.model
            self.network_cache[str(input_shapes)] = model
        else:
            model = self.network_cache[str(input_shapes)]

        compiled_model = self.runtime.backend.compile_model(model, self.runtime.backend_name)
        is_bfloat16 = any(parameter.get_output_element_type(0) == Type.bf16 for parameter in self.parameters)
        is_float8 = any(parameter.get_output_element_type(0) == Type.f8e4m3 or parameter.get_output_element_type(0) == Type.f8e5m2 for parameter in self.parameters)
        if is_bfloat16 or is_float8:
            input_values = self.convert_to_tensors(input_values)
        request = compiled_model.create_infer_request()
        result_buffers = request.infer(dict(zip(param_names, input_values)))
        """Note: other methods to get result_buffers from request
           First call infer with no return value:
            request.infer(dict(zip(param_names, input_values)))
           Now use any of following options:
            result_buffers = [request.get_tensor(n).data for n in request.outputs]
            result_buffers = [request.get_output_tensor(i).data for i in range(len(request.outputs))]
            result_buffers = [t.data for t in request.output_tensors]
        """
        # # Since OV overwrite result data type we have to convert results to the original one.
        original_dtypes = [get_onnx_test_dtype(result.get_output_element_type(0)) for result in self.results]
        converted_buffers = self.convert_buffers(result_buffers, original_dtypes)
        return converted_buffers
