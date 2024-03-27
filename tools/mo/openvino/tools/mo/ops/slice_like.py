# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, is_fully_defined
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class SliceLike(Op):
    op = 'slice_like'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        assert 'axes' in attrs, 'Please set mandatory `axes` attribute for `slice_like` operation'
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.infer,
        }, attrs)

    @staticmethod
    def infer(node):
        input_shape = node.in_port(0).data.get_shape()
        input_value = node.in_port(0).data.get_value()
        shape_like = node.in_port(1).data.get_shape()

        new_shape = shape_array(input_shape.copy())
        if node.axes is not None:
            node.axes = sorted([get_canonical_axis_index(input_shape, i) for i in node.axes])
            for i in node.axes:
                new_shape[i] = shape_like[i]
        else:
            assert input_shape.size == shape_like.size,\
                'Input shape ranks are inconsistent: {} and {}'.format(input_shape.size, shape_like.size)
            node.axes = int64_array(range(shape_like.size))
            new_shape = shape_like.copy()
        node.out_port(0).data.set_shape(new_shape)

        if input_value is not None and is_fully_defined(new_shape):
            out_value = np.copy(input_value)

            slice_indexes = []
            for s in out_value.shape:
                slice_indexes.append(slice(0, s))

            for axis in node.axes:
                slice_indexes[axis] = slice(0, new_shape[axis])
                out_value = out_value[tuple(slice_indexes)]
            node.out_port(0).data.set_value(out_value)
