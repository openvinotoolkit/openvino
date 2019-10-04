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

import logging as log

# Concat infer : N - number of inputs to concat
#                axis - dimension number for tensors concatenation
import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.ops.op import PermuteAttrs


def concat_infer(node):
    if not node.has('axis'):
        N = node.N
        axis_input = node.in_node(N)
        if axis_input.has_valid('value') and axis_input.value.size == 1:
            node['axis'] = axis_input.value.item()
            node.graph.remove_edge(axis_input.node, node.node)  # TODO add skip attribute instead of deleting
        else:
            return
    else:
        N = len(node.in_nodes())

    shapes = [node.in_node(i).shape for i in range(N)]
    if any(s is None for s in shapes):
        return

    shape = np.array(shapes[0])

    axis = get_canonical_axis_index(shape, node.axis)
    node.axis = axis

    mask = np.zeros_like(shape, dtype=np.bool)
    mask[axis] = True  # pylint: disable=unsupported-assignment-operation
    not_mask = np.logical_not(mask)  # pylint: disable=assignment-from-no-return
    for s in shapes[1:]:
        if np.all(shape[not_mask] == s[not_mask]):  # TODO handle -1 in a special way
            shape[mask] += s[mask]
        else:
            log.error('Concat input shapes do not match')
            return

    node.out_node(0).shape = shape
    if len(shape) != 4:
        # exclude it from NHWC to NCHW convertion
        if 'axis' in node.dim_attrs:
            node.dim_attrs.remove('axis')

    PermuteAttrs.create_permute_attrs(node, attrs=[('axis','input:0')])

    values = [node.in_node(i).value for i in range(N)]
    if any(v is None for v in values):
        return

    node.out_node(0).value = np.array(np.concatenate(values, axis=node.axis), dtype=values[0].dtype)
    node.out_node(0).shape = np.array(node.out_node(0).value.shape, dtype=np.int64)




def tf_pack_infer(node):
    # Constant path is supported only
    values = [node.in_node(i).value for i in range(node.N)]
    if any(v is None for v in values):
        return
    node.out_node().value = np.stack(values, node.axis)
    node.out_node().shape = np.array(node.out_node().value.shape, dtype=np.int64)

