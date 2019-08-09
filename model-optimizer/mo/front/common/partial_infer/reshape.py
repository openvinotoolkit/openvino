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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.perm_inputs import PermuteInputs
from mo.utils.error import Error


def tf_reshape_shape_infer(node):
    # TODO Make sure that all -1 are handled correctly
    # We cannot simply copy shape argument to the output,
    # because if -1 appears, it should be substituted by a real
    # value from input shape if input shape is completely defined.
    if node.in_node(0).shape is None:
        return None

    assert len(node.in_nodes()) == 2, 'The Reshape operation {} must have 2 inputs'.format(node.name)

    input_shape = node.in_port(0).data.get_shape()
    reshape_output = node.in_port(1).data.get_value()

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
        raise Error("Number of elements in input {} and output {} of reshape node {} mismatch"
                    "".format(input_shape, output_shape, node.name))

    PermuteInputs().set_input_permutation(node.in_node(1), node, 'output:0', 'shape')

    output_shape = int64_array(output_shape)
    return output_shape
