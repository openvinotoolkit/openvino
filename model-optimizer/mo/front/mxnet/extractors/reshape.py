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


def reshape_ext(attr):
    dim = attr.tuple("shape", int, None)
    node_attrs = {
        'type': 'Reshape',
        'axis': 0,
        'num_axes': -1,
        'dim': np.array(dim),
        'infer': lambda node: single_output_infer(node, tf_reshape_shape_infer)
    }
    return node_attrs
