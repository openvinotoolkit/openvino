# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.common.layout import get_batch_dim, get_height_dim, get_width_dim, shape_for_layout
from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, dynamic_dimension_value, \
    undefined_shape_of_rank
from openvino.tools.mo.front.extractor import attr_getter, bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class RegionYoloOp(Op):
    op = 'RegionYolo'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'in_ports_count': 1,
            'out_ports_count': 1,
            'reverse_infer': self.reverse_infer,
            'infer': RegionYoloOp.regionyolo_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'coords',
            'classes',
            'num',
            'axis',
            'end_axis',
            'do_softmax',
            'anchors',
            'mask'
        ]

    def backend_attrs(self):
        return [
            'coords',
            'classes',
            'num',
            'axis',
            'end_axis',
            ('do_softmax', lambda node: bool_to_str(node, 'do_softmax')),
            ('anchors', lambda node: attr_getter(node, 'anchors')),
            ('mask', lambda node: attr_getter(node, 'mask'))
        ]

    @staticmethod
    def regionyolo_infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        axis = get_canonical_axis_index(input_shape, node.axis)
        end_axis = get_canonical_axis_index(input_shape, node.end_axis)
        node.axis = axis
        node.end_axis = end_axis
        if node.do_softmax:
            dims_to_flatten = input_shape[axis: end_axis + 1]
            if is_fully_defined(dims_to_flatten):
                flat_dim = np.ma.prod(dims_to_flatten)
            else:
                flat_dim = dynamic_dimension_value
            node.out_port(0).data.set_shape([*input_shape[:axis], flat_dim, *input_shape[end_axis + 1:]])
        else:
            layout = node.graph.graph['layout']
            assert len(layout) == 4

            node.out_port(0).data.set_shape(shape_for_layout(layout,
                                                             batch=input_shape[get_batch_dim(layout, 4)],
                                                             features=(node.classes + node.coords + 1) * len(node.mask),
                                                             height=input_shape[get_height_dim(layout, 4)],
                                                             width=input_shape[get_width_dim(layout, 4)]))

    @staticmethod
    def reverse_infer(node):
        if node.in_port(0).data.get_shape() is None:
            node.in_port(0).data.set_shape(undefined_shape_of_rank(4))
