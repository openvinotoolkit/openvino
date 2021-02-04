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

from mo.front.common.partial_infer.utils import get_shape_from_slice
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error
from mo.utils.utils import array_to_str


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
        begin = node.in_port(1).data.get_value()
        end = node.in_port(2).data.get_value()
        if begin is None or end is None:
            raise Error('StridedSlice operation supports only constant begin and end inputs')

        if len(node.in_nodes()) > 3:
            strides = node.in_port(3).data.get_value()
            if strides is None:
                raise Error('StridedSlice operation supports only constant strides input')
        else:
            strides = np.ones_like(begin)

        shape = node.in_port(0).data.get_shape()
        value = node.in_port(0).data.get_value()
        input_rank = len(shape)
        assert len(begin) == len(end) == len(strides), 'begin, end, and strides must be of the same length'

        if shape is None or any([x < 0 for x in shape]):
            return

        # extend all masks to match initial slice_rank
        extend_mask = lambda mask, val=0: np.append(mask, [val] * (len(begin) - len(mask))).astype(int)
        new_axis_mask = extend_mask(node.new_axis_mask)
        shrink_axis_mask = extend_mask(node.shrink_axis_mask)
        begin_mask = extend_mask(node.begin_mask, 1)
        end_mask = extend_mask(node.end_mask, 1)  # todo: differs from case when we unroll ellipsis
        # no need to extend ellipsis

        # unroll ellipsis
        if np.any(node.ellipsis_mask):
            i = np.nonzero(node.ellipsis_mask)
            assert len(i[0]) == 1, 'only one nonzero value in ellipsis_mask is allowed'
            ellipsis_start = i[0][0]
            # since we don't expect values in begin, end values and take all range of values along ellipsis_start axis
            begin_mask[ellipsis_start] = 0
            end_mask[ellipsis_start] = 0

            num = input_rank - len(begin) + np.count_nonzero(node.new_axis_mask[ellipsis_start:])
            unroll_ellipsis = lambda mask, val=0: np.insert(mask, ellipsis_start + 1, [val] * num).astype(int)

            new_axis_mask = unroll_ellipsis(new_axis_mask)
            shrink_axis_mask = unroll_ellipsis(shrink_axis_mask)
            begin_mask, end_mask = unroll_ellipsis(begin_mask), unroll_ellipsis(end_mask)
            begin, end, strides = unroll_ellipsis(begin), unroll_ellipsis(end), unroll_ellipsis(strides, 1)

        # from now slices are without ellipsis
        slice_rank = len(begin)
        slices = [[]] * slice_rank
        in_idx = 0  # index along input tensor shapes, note that input_rank not necessary is equal to slice_rank
        for i in range(slice_rank):
            if new_axis_mask[i]:
                slices[i] = np.newaxis
            elif shrink_axis_mask[i]:
                slices[i] = int(begin[i])
                if slices[i] < 0:
                    slices[i] += int(shape[in_idx])
            else:
                start, stop = begin[i], end[i]
                if not begin_mask[i]:  # if begin, and end are not specified take whole range
                    start = 0 if strides[i] > 0 else -1
                if not end_mask[i]:
                    stop = shape[in_idx] if strides[i] > 0 else -shape[in_idx] - 1
                slices[i] = slice(start, stop, strides[i])
            in_idx += 1 if not new_axis_mask[i] else 0

        if value is not None:
            node.out_port(0).data.set_value(value[tuple(slices)])
        else:
            node.out_port(0).data.set_shape(get_shape_from_slice(shape, slices))

        # normalize slices attr which is used by ConvertGroupedStridedSlice
        in_idx = 0
        for i in range(slice_rank):
            if new_axis_mask[i]:
                slices[i] = slice(0, 1, 1)
            elif shrink_axis_mask[i]:
                slices[i] = slice(slices[i], slices[i] + 1, strides[i])
            if not new_axis_mask[i]:
                slices[i] = slice(*slices[i].indices(shape[in_idx]))  # will convert negative indices
                in_idx += 1
        node['slices'] = np.array(slices)

        node['force_precision_in_ports'] = {port: 'int64' for port in range(1, len(node.in_nodes()))}
