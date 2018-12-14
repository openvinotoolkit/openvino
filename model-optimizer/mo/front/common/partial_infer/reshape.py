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

import logging as log

import numpy as np

from mo.ops.op import PermuteAttrs


def tf_reshape_shape_infer(node):
    # TODO Make sure that all -1 are handled correctly
    # We cannot simply copy shape argument to the output,
    # because if -1 appears, it should be substituted by a real
    # value from input shape if input shape is completely defined.
    if node.in_node(0).shape is None:
        return None

    input_shape = node.in_node(0).shape
    reshape_output = node.in_node(1).value if len(node.in_nodes()) > 1 else node.dim

    # In case if Reshape operation was created with two inputs and dim attr wasn't set, we set in automatically
    if not node.has_valid('dim'):
        node['dim'] = np.array(reshape_output, dtype=np.int64)

    if node.in_node(0).shape is None:
        return None
    total = 1
    for index, i in enumerate(input_shape):
        total *= i

    res = 1
    for index, x in enumerate(reshape_output):
        if x == 0:
            res *= input_shape[index]
        elif x != -1:
            res *= x

    new_dim = total // res
    output_shape = []
    for index, x in enumerate(reshape_output):
        if x == 0:
            output_shape.append(input_shape[index])
        elif x == -1:
            output_shape.append(new_dim)
        else:
            output_shape.append(x)

    out_shape_total = 1
    for index, i in enumerate(output_shape):
        assert i != -1
        out_shape_total *= i

    if total != out_shape_total:
        log.error(
            "Number of elements in input {} and output {} of reshape node {} mismatch".format(input_shape, output_shape,
                                                                                              node.name))
        return None

    PermuteAttrs.create_permute_attrs(node, attrs=[('dim', 'output:0')])

    return np.array(output_shape, dtype=np.int64)
