"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from mo.front.common.partial_infer.const import tf_const_infer
from onnx.numpy_helper import to_array  # TODO eliminate direct onnx dependency


def onnx_const_ext(node):
    value = to_array(node.pb_init)
    result = {
        'data_type': value.dtype,
        'shape': np.array(value.shape),
        'value': value,
        'infer': tf_const_infer
    }
    return result
