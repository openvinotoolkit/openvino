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


def tf_expand_dims_infer(node):
    input_node = node.in_nodes()[0]
    output_node = node.out_node()
    if input_node.shape is None:
        return

    # TensorFlow style with dynamic input
    if len(node.in_nodes()) > 1:
        axis_node = node.in_nodes()[1]
        if isinstance(axis_node.value, np.ndarray) and axis_node.value.size > 1:
            log.error("ExpandDims operation : axis should be scalar")
            return
        expand_axis = axis_node.value.item()
        node.graph.remove_edge(axis_node.id, node.id)
    else:
        if not node.has_valid('expand_axis'):
            log.error("ExpandDims axis is not defined")
            return
        expand_axis = node.expand_axis

    if expand_axis is None:
        return

    output_node.shape = np.insert(input_node.shape, expand_axis, [1])
    # convert data type of the shape to int64 explicitly
    output_node.shape = output_node.shape.astype(np.int64)
    if input_node.value is not None:
        output_node.value = np.array(np.reshape(input_node.value, output_node.shape))

    node['axis'] = 0
    node['num_axes'] = -1
    node['dim'] = output_node.shape

    PermuteAttrs.create_permute_attrs(node, attrs=[('axis','output:0'), ('dim','output:0')])

