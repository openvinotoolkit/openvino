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
    def __init__(self, ng_model_function, device='CPU'):  # type: (List[Function], str) -> None
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
    def is_compatible(cls,
                      model,  # type: ModelProto
                      device='CPU',  # type: Text
                      **kwargs  # type: Any
                      ):  # type: (...) -> bool
        # Return whether the model is compatible with the backend.
        try:
            import_onnx_model(model)
            return True
        except Exception:
            return False

    @classmethod
    def prepare(cls,
                onnx_model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> OpenVinoOnnxBackendRep
        onnx.checker.check_model(onnx_model)
        super().prepare(onnx_model, device, **kwargs)
        ng_model_function = import_onnx_model(onnx_model)
        return OpenVinoOnnxBackendRep(ng_model_function, cls.backend_name)

    @classmethod
    def run_model(cls,
                  model,  # type: ModelProto
                  inputs,  # type: Any
                  device='CPU',  # type: Text
                  **kwargs  # type: Any
                  ):  # type: (...) -> Tuple[Any, ...]
        cls.prepare(model, device, **kwargs).run()

    @classmethod
    def run_node(cls,
                 node,  # type: NodeProto
                 inputs,  # type: Any
                 device='CPU',  # type: Text
                 outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
                 **kwargs  # type: Dict[Text, Any]
                 ):  # type: (...) -> Optional[Tuple[Any, ...]]
        """Prepare and run a computation on an ONNX node."""
        # default values for input/output tensors
        input_tensor_types = [np_dtype_to_tensor_type(node_input.dtype) for node_input in inputs]
        output_tensor_types = [onnx.TensorProto.FLOAT for idx in range(len(node.output))]
        output_tensor_shapes = [()]  # type: List[Tuple[int, ...]]

        if outputs_info is not None:
            output_tensor_types = [np_dtype_to_tensor_type(dtype) for (dtype, shape) in
                                   outputs_info]
            output_tensor_shapes = [shape for (dtype, shape) in outputs_info]

        input_tensors = [make_tensor_value_info(name, tensor_type, value.shape)
                         for name, value, tensor_type in zip(node.input, inputs,
                                                             input_tensor_types)]
        output_tensors = [make_tensor_value_info(name, tensor_type, shape)
                          for name, shape, tensor_type in zip(node.output, output_tensor_shapes,
                                                              output_tensor_types)]

        graph = make_graph([node], 'compute_graph', input_tensors, output_tensors)
        model = make_model(graph, producer_name='OpenVinoOnnxBackend')
        if 'opset_version' in kwargs:
            model.opset_import[0].version = kwargs['opset_version']
        return cls.prepare(model, device).run(inputs)

    @classmethod
    def supports_device(cls, device):  # type: (Text) -> bool
        """
        Checks whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        return True



# class NgraphBackend(Backend):
#     """Takes an ONNX model with inputs, perform a computation, and then return the output."""
#
#     # The requested (nGraph) backend to be used instead of hardcoded by ONNX test Runner.
#     backend_name = 'CPU'  # type: str
#
#     _ngraph_onnx_device_map = [
#         # (<ngraph_backend_name>, <onnx_device_name>)
#         ('CPU', 'CPU'),
#         ('GPU', 'CUDA'),
#         ('INTELGPU', 'CPU'),
#         ('INTERPRETER', 'CPU'),
#         ('ARGON', 'CPU'),
#         ('NNP', 'CPU'),
#         ('PlaidML', 'CPU'),
#         ('IE:CPU', 'CPU'),
#     ]
#
#     @classmethod
#     def prepare(cls, onnx_model, device='CPU', **kwargs):
#         # type: (onnx.ModelProto, str, Dict) -> NgraphBackendRep
#         """Prepare backend representation of ONNX model."""
#         super(NgraphBackend, cls).prepare(onnx_model, device, **kwargs)
#         ng_model_function = import_onnx_model(onnx_model)
#         return NgraphBackendRep(ng_model_function, cls.backend_name)
#
#     @classmethod
#     def _get_onnx_device_name(cls, ngraph_device_name):  # type: (str) -> Optional[str]
#         return next((onnx_device for (ng_device, onnx_device) in cls._ngraph_onnx_device_map
#                      if ngraph_device_name == ng_device), None)
#
#     @classmethod
#     @lru_cache(maxsize=16)
#     def supports_ngraph_device(cls, ngraph_device_name):  # type: (str) -> bool
#         """Check whether particular nGraph device is supported by current nGraph library.
#
#         :param ngraph_device_name: Name of nGraph device.
#         :return: True if current nGraph library supports ngraph_device_name.
#         """
#         try:
#             ng.runtime(backend_name=ngraph_device_name)
#         except RuntimeError as e:
#             # Catch error raised when backend isn't available:
#             # 'Backend {ngraph_device_name} not found in registered backends'
#             if str(ngraph_device_name) in str(e) and 'not found' in str(e):
#                 return False
#             else:
#                 raise e
#         return True
#
#     @classmethod
#     def supports_device(cls, onnx_device_name):  # type: (str) -> bool
#         """Check whether the requested nGraph backend supports a particular ONNX device.
#
#         During running ONNX backend tests this function is called on each item of ONNX defined
#         devices list. Currently this list is hardcoded and contains only two entries:
#          ('CPU', 'CUDA'). In order to check whether the requested nGraph backend stored as
#          NgraphBackend class variable we have to map its name into ONNX device namespace and then
#          verify whether the current version of nGraph library supports it.
#
#         :param onnx_device_name: One of ONNX defined devices.
#         :return: True if ONNX device is supported, otherwise False.
#         """
#         requested_backend_name_mapped_to_onnx_device = cls._get_onnx_device_name(cls.backend_name)
#         # Check whether:
#         # 1. There is mapping between onnx_device_name and requested nGraph backend to run tests on.
#         # 2. Current nGraph version supports requested backend.
#         return (onnx_device_name == requested_backend_name_mapped_to_onnx_device
#                 and cls.supports_ngraph_device(cls.backend_name))
#
#     @classmethod
#     def run_model(cls, onnx_model, inputs, device='CPU', **kwargs):
#         # type: (onnx.ModelProto, List[np.ndarray], str, Dict) -> List[Any]
#         """Prepare and run a computation on an ONNX model."""
#         return cls.prepare(onnx_model, device, **kwargs).run(inputs)
#
#     @classmethod
#     def run_node(cls,
#                  node,  # type: onnx.NodeProto
#                  inputs,  # type: List[np.ndarray]
#                  device='CPU',  # type: Text
#                  outputs_info=None,  # type: Optional[Sequence[Tuple[np.dtype, Tuple[int, ...]]]]
#                  **kwargs  # type: Any
#                  ):  # type: (...) -> List[Any]
#         """Prepare and run a computation on an ONNX node."""
#         # default values for input/output tensors
#         input_tensor_types = [np_dtype_to_tensor_type(node_input.dtype) for node_input in inputs]
#         output_tensor_types = [onnx.TensorProto.FLOAT for idx in range(len(node.output))]
#         output_tensor_shapes = [()]  # type: List[Tuple[int, ...]]
#
#         if outputs_info is not None:
#             output_tensor_types = [np_dtype_to_tensor_type(dtype) for (dtype, shape) in
#                                    outputs_info]
#             output_tensor_shapes = [shape for (dtype, shape) in outputs_info]
#
#         input_tensors = [make_tensor_value_info(name, tensor_type, value.shape)
#                          for name, value, tensor_type in zip(node.input, inputs,
#                                                              input_tensor_types)]
#         output_tensors = [make_tensor_value_info(name, tensor_type, shape)
#                           for name, shape, tensor_type in zip(node.output, output_tensor_shapes,
#                                                               output_tensor_types)]
#
#         graph = make_graph([node], 'compute_graph', input_tensors, output_tensors)
#         model = make_model(graph, producer_name='NgraphBackend')
#         if 'opset_version' in kwargs:
#             model.opset_import[0].version = kwargs['opset_version']
#         return cls.prepare(model, device).run(inputs)
#
#

#
#     def _get_ngraph_device_name(self, onnx_device):  # type: (str) -> str
#         return 'GPU' if onnx_device == 'CUDA' else onnx_device
#
#
prepare = OpenVinoOnnxBackend.prepare
run_model = OpenVinoOnnxBackend.run_model
run_node = OpenVinoOnnxBackend.run_node
supports_device = OpenVinoOnnxBackend.supports_device
