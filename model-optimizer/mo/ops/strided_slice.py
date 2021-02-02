"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error
from mo.utils.utils import array_to_str
from mo.ops.slice import get_shape_from_slice


class StridedSlice(Op):
    op = 'StridedSlice'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': 'StridedSlice',
            'version': 'opset1',
            'in_ports_count': 4,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)
        assert 'new_axis_mask' in attrs, "Attribute 'new_axis_mask' of the StridedSlice node is not given."
        assert 'shrink_axis_mask' in attrs, "Attribute 'shrink_axis_mask' of the StridedSlice node is not given."
        assert 'ellipsis_mask' in attrs, "Attribute 'ellipsis_mask' of the StridedSlice node is not given."
        assert 'begin_mask' in attrs, "Attribute 'begin_mask' of the StridedSlice node is not given."
        assert 'end_mask' in attrs, "Attribute 'end_mask' of the StridedSlice node is not given."

    def backend_attrs(self):
        al = list()

        def convert(attr):
            return lambda node: array_to_str(node, attr)

        for a in list(['new_axis_mask', 'shrink_axis_mask', 'ellipsis_mask', 'begin_mask', 'end_mask']):
            al.append((a, convert(a)))
        return al

    @staticmethod
    def infer(node: Node):
        # FW assures that begin, end, and strides are of the same length th
        tf_strided_slice_infer(node)


def tf_strided_slice_infer(node):
    if node.in_node(1).value is None or node.in_node(2).value is None:
        raise Error('Strided slice layer supports only constant begin and end inputs')
    begin_id = node.in_port(1).data.get_value()
    end_id = node.in_port(2).data.get_value()

    if len(node.in_nodes()) > 3:
        if node.in_port(3).data.get_value() is None:
            raise Error('Strided slice layer supports only constant stride input')
        strides = node.in_port(3).data.get_value()
    else:
        strides = np.ones_like(begin_id)
    shape = node.in_node(0).shape
    value = node.in_port(0).data.get_value()
    input_rank = len(shape)

    if shape is None or any([x < 0 for x in shape]):
        return

    assert len(begin_id) == len(end_id) == len(strides), 'begin, end, and strides must be of the same length'

    extend_mask = lambda mask: np.append(mask, [0] * (len(begin_id) - len(mask)))
    new_axis_mask = extend_mask(node.new_axis_mask)
    shrink_axis_mask = extend_mask(node.shrink_axis_mask)
    begin_mask = extend_mask(node.begin_mask)
    end_mask = extend_mask(node.end_mask)

    # unroll ellipsis
    if np.any(node.ellipsis_mask):
        i = np.nonzero(node.ellipsis_mask)
        assert len(i[0]) == 1, 'only one nonzero value in ellipsis_mask is allowed'
        ellipsis_start = i[0][0]
        num_inserts = input_rank - len(begin_id) + np.count_nonzero(node.new_axis_mask[ellipsis_start:])

        # since we don't unse begin, end value
        begin_mask[ellipsis_start] = 0
        end_mask[ellipsis_start] = 0
        new_axis_mask = np.insert(new_axis_mask, ellipsis_start + 1, [0] * num_inserts)
        shrink_axis_mask = np.insert(shrink_axis_mask, ellipsis_start + 1, [0] * num_inserts)
        begin_mask = np.insert(begin_mask, ellipsis_start + 1, [0] * num_inserts)
        end_mask = np.insert(end_mask, ellipsis_start + 1, [0] * num_inserts)

        begin_id = np.insert(end_id, ellipsis_start + 1, [0] * num_inserts)
        end_id = np.insert(end_id, ellipsis_start + 1, [0] * num_inserts)
        strides = np.insert(strides, ellipsis_start + 1, [1] * num_inserts)

    # from now slices are without ellipsis
    dims = len(begin_id)
    slice_idx = [[]] * dims
    in_idx = 0
    for i in range(dims):
        if new_axis_mask[i]:
            slice_idx[i] = np.newaxis
        elif shrink_axis_mask[i]:
            begin = begin_id[in_idx]
            if begin < 0:
                begin += shape[in_idx]
            slice_idx[i] = int(begin)
        else:
            begin = begin_id[in_idx]
            end = end_id[in_idx]
            if not begin_mask[i]:
                begin = 0 if strides[in_idx] > 0 else -1
            if not end_mask[i]:
                end = shape[in_idx] if strides[in_idx] > 0 else -shape[in_idx] - 1
            slice_idx[i] = slice(begin, end, strides[in_idx])
        in_idx += 1 if not new_axis_mask[i] else 0

    if value is not None:
        node.out_port(0).data.set_value(value[tuple(slice_idx)])
    else:
        node.out_port(0).data.set_shape(get_shape_from_slice(shape, slice_idx))

    in_idx = 0
    for i in range(dims):
        if new_axis_mask[i]:
            slice_idx[i] = slice(0, 1, 1)
        elif shrink_axis_mask[i]:
            slice_idx[i] = slice(slice_idx[i], slice_idx[i] + 1, strides[i])
        if not new_axis_mask[i]:
            slice_idx[i] = slice(*slice_idx[i].indices(shape[in_idx]))  # will convert negative indices
            in_idx += 1
    node['slices'] = np.array(slice_idx)

    node['force_precision_in_ports'] = {port: 'int64' for port in range(1, len(node.in_nodes()))}
