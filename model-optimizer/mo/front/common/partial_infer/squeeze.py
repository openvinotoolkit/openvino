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


def tf_squeeze_infer(node):
    if node.squeeze_dims is None:
        # TODO: implement; there is no implementation now because no test
        return
    real_squeeze_dims = []
    shape = node.in_node().shape
    if shape is None:
        return
    # UGLY
    shape = shape.copy()
    for n in node.squeeze_dims:
        if shape[n] == 1:
            real_squeeze_dims.append(n)
    shape = np.delete(shape, real_squeeze_dims)
    node.out_node().shape = shape
    if node.in_node().value is not None:
        node.out_node().value = np.array(np.reshape(node.in_node().value, shape))
