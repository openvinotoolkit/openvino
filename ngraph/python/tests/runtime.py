# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""Provide a layer of abstraction for an OpenVINO runtime environment."""
import logging
from typing import Dict, List, Union

import numpy as np
from openvino.inference_engine import IECore, IENetwork

from ngraph.exceptions import UserInputError
from ngraph.impl import Function, Node, PartialShape
from ngraph.utils.types import NumericData, get_shape, get_dtype

import tests

log = logging.getLogger(__name__)


def runtime(backend_name: str = "CPU") -> "Runtime":
    """Create a Runtime object (helper factory)."""
    return Runtime(backend_name)


def get_runtime():
    """Return runtime object."""
    return runtime(backend_name=tests.BACKEND_NAME)


def convert_i64_to_i32(cnn_network: IENetwork) -> None:
    for cnn_input in cnn_network.input_info:
        if cnn_network.input_info[cnn_input].precision == "I64":
            cnn_network.input_info[cnn_input].precision = "I32"


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

    def __call__(self, *input_values: NumericData) -> List[NumericData]:
        """Run computation on input values and return result."""
        input_values = [np.array(input_value) for input_value in input_values]
        input_shapes = [get_shape(input_value) for input_value in input_values]

        param_names = [param.friendly_name for param in self.parameters]

        if self.network_cache.get(str(input_shapes)) is None:
            capsule = Function.to_capsule(self.function)
            cnn_network = IENetwork(capsule)
            if self.function.is_dynamic():
                cnn_network.reshape(dict(zip(param_names, input_shapes)))
            # Convert inputs of the network from I64 to I32
            convert_i64_to_i32(cnn_network)
            self.network_cache[str(input_shapes)] = cnn_network
        else:
            cnn_network = self.network_cache[str(input_shapes)]

        executable_network = self.runtime.backend.load_network(cnn_network, self.runtime.backend_name)

        # Input validation
        if len(input_values) != len(self.parameters):
            raise UserInputError(
                "Expected %s parameters, received %s.", len(self.parameters), len(input_values)
            )
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

        # Since OV overwrite result data type we have to convert results to the original one.
        original_dtypes = [get_dtype(result.get_output_element_type(0)) for result in self.results]
        result_buffers = [blob.buffer for blob in request.output_blobs.values()]
        converted_buffers = [buffer.astype(original_dtype) for buffer, original_dtype in
                             zip(result_buffers, original_dtypes)]
        return converted_buffers
