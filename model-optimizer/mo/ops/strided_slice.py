# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

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
        for mask_name in StridedSlice.get_mask_names():
            assert mask_name in attrs, 'Attribute {} of the StridedSlice node is not given.'.format(mask_name)

    @staticmethod
    def get_mask_names():
        return ['begin_mask', 'end_mask', 'new_axis_mask', 'shrink_axis_mask', 'ellipsis_mask']

    def backend_attrs(self):
        al = list()

        def convert(attr):
            return lambda node: array_to_str(node, attr)

        for a in StridedSlice.get_mask_names():
            al.append((a, convert(a)))
        return al

    @staticmethod
    def infer(node: Node):
        begin, end, strides = StridedSlice.validate_inputs_and_get_args(node)

        StridedSlice.align_mask_with_slice_rank(node, len(begin))

        data_shape = node.in_port(0).data.get_shape()
        data_value = node.in_port(0).data.get_value()
        slices = StridedSlice.get_slices(node, data_shape, begin, end, strides)

        if data_value is not None:
            node.out_port(0).data.set_value(data_value[tuple(slices)])
        else:
            node.out_port(0).data.set_shape(get_shape_from_slice(data_shape, slices))

        node['slices'] = slices
        node['force_precision_in_ports'] = {port: 'int64' for port in range(1, len(node.in_nodes()))}

        # StridedSliceNormalizer inserts nodes that change original begin, end, and strides data nodes
        # and since input permutations are stored in data nodes we end up having permutations
        # in the wrong place of the graph.
        # Therefore PermuteInputs will be set after StridedSliceNormalizer.

    @staticmethod
    def get_slices(node: Node, data_shape: Tuple, begin: np.array, end: np.array, strides: np.array) -> List:
        input_rank = len(data_shape)
        slice_rank = len(begin)
        # from now slices are without ellipsis
        slices = [[]] * slice_rank
        in_idx = 0  # index along input tensor shapes, note that input_rank not necessary is equal to slice_rank
        for i in range(slice_rank):
            if node.new_axis_mask[i]:
                slices[i] = np.newaxis
            elif node.shrink_axis_mask[i]:
                slices[i] = int(begin[i])
                if slices[i] < 0:  # need for ConvertGroupedStridedSlice
                    slices[i] += int(data_shape[in_idx])
            elif node.ellipsis_mask[i]:
                slices[i] = ...
                in_idx += input_rank - slice_rank + np.count_nonzero(node.new_axis_mask)
            else:
                start, stop = begin[i], end[i]
                if not node.begin_mask[i]:  # if begin, and end are not specified take the whole range
                    start = None
                if not node.end_mask[i]:
                    stop = None
                slices[i] = slice(start, stop, strides[i])
            in_idx += 1 if not node.new_axis_mask[i] else 0
        return slices

    @staticmethod
    def align_mask_with_slice_rank(node: Node, slice_rank: int):
        # align masks sizes with slice_rank (not confuse with extending, mask_aligment != mask_extending)
        for mask_name in StridedSlice.get_mask_names():
            num_insertations = slice_rank - len(node[mask_name])
            val = 0 if mask_name not in ['begin_mask', 'end_mask'] else 1  # extend with ones only for begin and end
            node[mask_name] = np.append(node[mask_name], [val] * num_insertations).astype(int)

    @staticmethod
    def validate_inputs_and_get_args(node: Node) -> (np.ndarray, np.ndarray, np.ndarray):
        node_name = node.soft_get('name', node.id)
        begin = node.in_port(1).data.get_value()
        end = node.in_port(2).data.get_value()

        if begin is None or end is None:
            raise Error(
                'StridedSlice operation for node {} supports only constant begin and end inputs'.format(node_name))

        if node.is_in_port_connected(3):
            strides = node.in_port(3).data.get_value()
            if strides is None:
                raise Error(
                    'StridedSlice operation for node {} supports only constant strides input'.format(node_name))
        else:
            strides = np.ones_like(begin)
        assert len(begin) == len(end) == len(strides), \
            'begin, end, and strides of StridedSlice node {} must be of the same length. Got insted:' \
            'begin = {}, end = {}, strides = {}'.format(node_name, begin, end, strides)
        return begin, end, strides
