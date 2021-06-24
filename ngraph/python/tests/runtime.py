# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Provide a layer of abstraction for an OpenVINO runtime environment."""

import logging
from typing import Dict, List, Union

import numpy as np
from openvino.inference_engine import IECore, IENetwork, Blob, DataPtr

from ngraph.exceptions import UserInputError
from ngraph.impl import Function, Node, PartialShape, Type
from ngraph.opset1.ops import result
from ngraph.utils.types import NumericData, get_shape, get_dtype

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


def _convert_inputs(cnn_network: IENetwork) -> None:
    """WA converts unsupported input images formats."""
    precision_map = {
        "FP64": "FP32",
        "U32": "I32",
    }

    for cnn_input in cnn_network.input_info:
        try:
            _precision = precision_map[cnn_network.input_info[cnn_input].precision]
            cnn_network.input_info[cnn_input].precision = _precision
        except KeyError:
            pass


def apply_ng_type(output: DataPtr, ng_type: Type):
    ng_ie_supported_type_map = {
        Type.boolean.get_type_name(): "BOOL",
        Type.f32.get_type_name(): "FP32",
        Type.i8.get_type_name(): "I8",
        Type.i32.get_type_name(): "I32",
        Type.u8.get_type_name(): "U8",
    }
    if ng_type.get_type_name() in ng_ie_supported_type_map:
        output.precision = ng_ie_supported_type_map[ng_type.get_type_name()]


class Runtime(object):
    """Represents an nGraph runtime environment."""

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name
        log.debug("Creating Inference Engine for %s" % backend_name)
        self.backend = IECore()
        assert backend_name in self.backend.available_devices, (
            'The requested device "' + backend_name + '" is not supported!'
        )

    def set_config(self, config: Dict[str, str]) -> None:
        """Set the inference engine configuration."""
        self.backend.set_config(config, device_name=self.backend_name)

    def __repr__(self) -> str:
        return "<Runtime: Backend='{}'>".format(self.backend_name)

    def computation(self, node_or_function: Union[Node, Function], *inputs: Node) -> "Computation":
        """Return a callable Computation object."""
        if isinstance(node_or_function, Node):
            ng_function = Function(node_or_function, inputs, node_or_function.name)
            return Computation(self, ng_function)
        elif isinstance(node_or_function, Function):
            return Computation(self, node_or_function)
        else:
            raise TypeError(
                "Runtime.computation must be called with an nGraph Function object "
                "or an nGraph node object an optionally Parameter node objects. "
                "Called with: %s",
                node_or_function,
            )


class Computation(object):
    """nGraph callable computation object."""

    def __init__(self, runtime: Runtime, ng_function: Function) -> None:
        self.runtime = runtime
        self.function = ng_function
        self.parameters = ng_function.get_parameters()
        self.results = ng_function.get_results()
        self.network_cache = {}

    def __repr__(self) -> str:
        params_string = ", ".join([param.name for param in self.parameters])
        return "<Computation: {}({})>".format(self.function.get_name(), params_string)

    def _get_ie_output_blob_name(self, outputs: Dict, ng_result: result) -> str:
        if len(self.results) == 1:
            return next(iter(outputs.keys()))
        else:
            prev_layer = ng_result.input(0).get_source_output()
            out_name = prev_layer.get_node().get_friendly_name()
            if prev_layer.get_node().get_output_size() != 1:
                out_name += "." + str(prev_layer.get_index())
            return out_name

    def _get_ie_output_blob_buffer(self, output_blobs: Dict[str, Blob], ng_result: result) -> np.ndarray:
        out_name = self._get_ie_output_blob_name(output_blobs, ng_result)
        out_blob = output_blobs[out_name]

        if out_blob.tensor_desc.layout == "SCALAR":
            return out_blob.buffer.reshape(())
        else:
            return out_blob.buffer

    def convert_buffers(self, source_buffers, target_dtypes):
        converted_buffers = []
        for i in range(len(source_buffers)):
            target_dtype = target_dtypes[i]
            # custom conversion for bf16
            if self.results[i].get_output_element_type(0) == Type.bf16:
                converted_buffers.append((source_buffers[i].view(np.uint32) >> 16).astype(np.uint16))
            else:
                converted_buffers.append(source_buffers[i].astype(target_dtype))
        return converted_buffers

    def __call__(self, *input_values: NumericData) -> List[NumericData]:
        """Run computation on input values and return result."""
        # Input validation
        if len(input_values) < len(self.parameters):
            raise UserInputError(
                "Expected %s params, received not enough %s values.", len(self.parameters), len(input_values)
            )
        # ignore not needed input values
        input_values = input_values[:len(self.parameters)]

        input_values = [np.array(input_value) for input_value in input_values]
        input_shapes = [get_shape(input_value) for input_value in input_values]

        param_names = [param.friendly_name for param in self.parameters]

        if self.network_cache.get(str(input_shapes)) is None:
            capsule = Function.to_capsule(self.function)
            cnn_network = IENetwork(capsule)
            if self.function.is_dynamic():
                cnn_network.reshape(dict(zip(param_names, input_shapes)))
            # Convert unsupported inputs of the network
            _convert_inputs(cnn_network)
            self.network_cache[str(input_shapes)] = cnn_network
        else:
            cnn_network = self.network_cache[str(input_shapes)]

        # set output blobs precission based on nG results
        for ng_result in self.results:
            ie_out_name = self._get_ie_output_blob_name(cnn_network.outputs, ng_result)
            apply_ng_type(cnn_network.outputs[ie_out_name], ng_result.get_output_element_type(0))

        executable_network = self.runtime.backend.load_network(cnn_network, self.runtime.backend_name)

        for parameter, input in zip(self.parameters, input_values):
            parameter_shape = parameter.get_output_partial_shape(0)
            input_shape = PartialShape(input.shape)
            if len(input.shape) > 0 and not parameter_shape.compatible(input_shape):
                raise UserInputError(
                    "Provided tensor's shape: %s does not match the expected: %s.",
                    input_shape,
                    parameter_shape,
                )

        request = executable_network.requests[0]
        request.infer(dict(zip(param_names, input_values)))

        # Set order of output blobs compatible with nG Function
        result_buffers = [self._get_ie_output_blob_buffer(request.output_blobs, result)
                          for result in self.results]

        # Since OV overwrite result data type we have to convert results to the original one.
        original_dtypes = [get_dtype(result.get_output_element_type(0)) for result in self.results]
        converted_buffers = self.convert_buffers(result_buffers, original_dtypes)
        return converted_buffers
