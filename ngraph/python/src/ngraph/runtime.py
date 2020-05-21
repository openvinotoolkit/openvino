# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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
"""Provide a layer of abstraction for the ngraph++ runtime environment."""
import logging
from typing import Dict, List, Union
from enum import Enum

import numpy as np

from ngraph.exceptions import UserInputError
from ngraph.impl import Function, Node, Shape, PartialShape, serialize, util
from ngraph.impl.runtime import Backend, Executable, Tensor
from ngraph.utils.types import NumericData, get_dtype

log = logging.getLogger(__name__)


class BackendMode(Enum):
    """DYNAMIC mode enables backend's wrapper which supports dynamic shapes."""

    STATIC = 0
    DYNAMIC = 1


def runtime(backend_name: str = "CPU", mode: BackendMode = BackendMode.STATIC) -> "Runtime":
    """Create a Runtime object (helper factory).

    Use signature to parameterize runtime as needed.
    """
    return Runtime(backend_name, mode)


class Runtime:
    """Represents the ngraph++ runtime environment."""

    def __init__(self, backend_name: str, mode: BackendMode = BackendMode.STATIC) -> None:
        self.backend_name = backend_name
        if mode == BackendMode.DYNAMIC:
            self.backend = Backend.create_dynamic(backend_name)
        else:
            self.backend = Backend.create(backend_name)

    def set_config(self, config: Dict[str, str]) -> None:
        """Set the backend configuration."""
        self.backend.set_config(config, "")

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
    """ngraph callable computation object."""

    def __init__(self, runtime: Runtime, ng_function: Function) -> None:
        self.runtime = runtime
        self.function = ng_function
        self.parameters = ng_function.get_parameters()
        self.results = ng_function.get_results()
        self.handle = self.runtime.backend.compile(self.function)

        self.tensor_views = []  # type: List[Tensor]
        for parameter in self.parameters:
            shape = parameter.get_shape()
            element_type = parameter.get_element_type()
            self.tensor_views.append(runtime.backend.create_tensor(element_type, shape))

        self.result_views = []  # type: List[Tensor]
        for result in self.results:
            element_type = result.get_element_type()
            if self.function.is_dynamic():
                output_pshape = result.get_output_partial_shape(0)
                output_tensor = runtime.backend.create_dynamic_tensor(element_type, output_pshape)
                self.result_views.append(output_tensor)
            else:
                output_shape = result.get_shape()
                output_tensor = runtime.backend.create_tensor(element_type, output_shape)
                self.result_views.append(output_tensor)

    def __repr__(self) -> str:
        params_string = ", ".join([param.name for param in self.parameters])
        return "<Computation: {}({})>".format(self.function.get_name(), params_string)

    def __call__(self, *input_values: NumericData) -> List[NumericData]:
        """Run computation on input values and return result."""
        for tensor_view, value in zip(self.tensor_views, input_values):
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            Computation._write_ndarray_to_tensor_view(value, tensor_view)

        if self.function.is_dynamic():
            self.handle.call_with_validate(self.result_views, self.tensor_views)
        else:
            self.handle.call(self.result_views, self.tensor_views)

        results = []
        for result_view in self.result_views:
            result = np.ndarray(result_view.shape, dtype=get_dtype(result_view.element_type))
            Computation._read_tensor_view_to_ndarray(result_view, result)
            results.append(result)

        return results

    def serialize(self, indent: int = 0) -> str:
        """Serialize function (compute graph) to a JSON string.

        :param indent: set indent of serialized output
        :return: serialized model
        """
        return serialize(self.function, indent)

    @staticmethod
    def _get_buffer_size(element_type: Tensor, element_count: int) -> int:
        return int((element_type.bitwidth / 8.0) * element_count)

    @staticmethod
    def _write_ndarray_to_tensor_view(value: np.ndarray, tensor_view: Tensor) -> None:
        tensor_view_dtype = get_dtype(tensor_view.element_type)
        if list(tensor_view.shape) != list(value.shape) and len(value.shape) > 0:
            raise UserInputError(
                "Provided tensor's shape: %s does not match the expected: %s.",
                list(value.shape),
                list(tensor_view.shape),
            )
        if value.dtype != tensor_view_dtype:
            log.warning(
                "Attempting to write a %s value to a %s tensor. Will attempt type conversion.",
                value.dtype,
                tensor_view.element_type,
            )
            value = value.astype(tensor_view_dtype)

        buffer_size = Computation._get_buffer_size(
            tensor_view.element_type, tensor_view.element_count
        )

        nparray = np.ascontiguousarray(value)
        tensor_view.write(util.numpy_to_c(nparray), buffer_size)

    @staticmethod
    def _read_tensor_view_to_ndarray(tensor_view: Tensor, output: np.ndarray) -> None:
        buffer_size = Computation._get_buffer_size(
            tensor_view.element_type, tensor_view.element_count
        )
        tensor_view.read(util.numpy_to_c(output), buffer_size)
