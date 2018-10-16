"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.common.partial_infer.elemental import single_output_infer
from mo.front.common.partial_infer.reshape import tf_reshape_shape_infer
from mo.front.onnx.extractors.utils import onnx_attr


def onnx_reshape_ext(node):
    ''' Extract ONNX Reshape op of different versions.
        Support both latest Reshape and Reshape-1.
        The first one has 2 arguments, Reshape-1 has one input and shape is coded in attribute.
    '''
    dim = onnx_attr(node, 'shape', 'ints', None)
    if dim is not None:
        dim = np.array(dim, dtype=np.int64)
    return {
        'type': 'Reshape',
        'dim': dim,
        'infer': lambda node: single_output_infer(node, tf_reshape_shape_infer,
                                                  lambda node: np.reshape(node.in_node().value,
                                                                          node.out_node().shape) if node.in_node().value is not None else None)
    }
