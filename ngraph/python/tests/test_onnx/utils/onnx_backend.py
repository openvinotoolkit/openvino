# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
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
"""
ONNX Backend implementation.

See ONNX documentation for details:
https://github.com/onnx/onnx/blob/master/docs/Implementing%20an%20ONNX%20backend.md
"""

import onnx

from onnx.helper import make_tensor_value_info, make_graph, make_model
from onnx.backend.base import Backend, BackendRep
from typing import Any, Dict, List, Optional, Sequence, Text, Tuple, Union

from ngraph.impl import Function
from tests.runtime import runtime
from tests.test_onnx.utils.onnx_helpers import np_dtype_to_tensor_type, import_onnx_model


class OpenVinoOnnxBackendRep(BackendRep):
    def __init__(self, ng_model_function, device="CPU"):  # type: (List[Function], str) -> None
        super().__init__()
        self.device = device
        self.ng_model_function = ng_model_function
        self.runtime = runtime(backend_name=self.device)
        self.computation = self.runtime.computation(ng_model_function)

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        """A handle which Backend returns after preparing to execute a model repeatedly."""
        return self.computation(*inputs)


class OpenVinoOnnxBackend(Backend):
    @classmethod
    def is_compatible(
        cls,
        model,  # type: ModelProto
        device="CPU",  # type: Text
        **kwargs  # type: Any
    ):  # type: (...) -> bool
        # Return whether the model is compatible with the backend.
        try:
            import_onnx_model(model)
            return True
        except Exception:
            return False

    @classmethod
    def prepare(
        cls,
        onnx_model,  # type: ModelProto
        device="CPU",  # type: Text
        **kwargs  # type: Any
    ):  # type: (...) -> OpenVinoOnnxBackendRep
        onnx.checker.check_model(onnx_model)
        super().prepare(onnx_model, device, **kwargs)
        ng_model_function = import_onnx_model(onnx_model)
        return OpenVinoOnnxBackendRep(ng_model_function, cls.backend_name)

    @classmethod
    def run_model(
        cls,
        model,  # type: ModelProto
        inputs,  # type: Any
        device="CPU",  # type: Text
        **kwargs  # type: Any
    ):  # type: (...) -> Tuple[Any, ...]
        cls.prepare(model, device, **kwargs).run()

    @classmethod
    def run_node(
        cls,
        node,  # type: NodeProto
        inputs,  # type: Any
        device="CPU",  # type: Text
        outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
        **kwargs  # type: Dict[Text, Any]
    ):  # type: (...) -> Optional[Tuple[Any, ...]]
        """Prepare and run a computation on an ONNX node."""
        # default values for input/output tensors
        input_tensor_types = [np_dtype_to_tensor_type(node_input.dtype) for node_input in inputs]
        output_tensor_types = [onnx.TensorProto.FLOAT for idx in range(len(node.output))]
        output_tensor_shapes = [()]  # type: List[Tuple[int, ...]]

        if outputs_info is not None:
            output_tensor_types = [np_dtype_to_tensor_type(dtype) for (dtype, shape) in outputs_info]
            output_tensor_shapes = [shape for (dtype, shape) in outputs_info]

        input_tensors = [
            make_tensor_value_info(name, tensor_type, value.shape)
            for name, value, tensor_type in zip(node.input, inputs, input_tensor_types)
        ]
        output_tensors = [
            make_tensor_value_info(name, tensor_type, shape)
            for name, shape, tensor_type in zip(node.output, output_tensor_shapes, output_tensor_types)
        ]

        graph = make_graph([node], "compute_graph", input_tensors, output_tensors)
        model = make_model(graph, producer_name="OpenVinoOnnxBackend")
        if "opset_version" in kwargs:
            model.opset_import[0].version = kwargs["opset_version"]
        return cls.prepare(model, device).run(inputs)

    @classmethod
    def supports_device(cls, device):  # type: (Text) -> bool
        """
        Checks whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        return True


prepare = OpenVinoOnnxBackend.prepare
run_model = OpenVinoOnnxBackend.run_model
run_node = OpenVinoOnnxBackend.run_node
supports_device = OpenVinoOnnxBackend.supports_device
