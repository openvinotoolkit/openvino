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
import numpy as np
import onnx
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from openvino.inference_engine import IECore

from ngraph.impl import Function


def np_dtype_to_tensor_type(data_type: np.dtype) -> int:
    """Return TensorProto type for provided numpy dtype.

    :param data_type: Numpy data type object.
    :return: TensorProto.DataType enum value for corresponding type.
    """
    return NP_TYPE_TO_TENSOR_TYPE[data_type]


def import_onnx_model(model: onnx.ModelProto) -> Function:
    onnx.checker.check_model(model)
    model_byte_string = model.SerializeToString()

    ie = IECore()
    ie_network = ie.read_network(model=model_byte_string, weights=b"", init_from_buffer=True)

    capsule = ie_network._get_function_capsule()
    ng_function = Function.from_capsule(capsule)
    return ng_function
