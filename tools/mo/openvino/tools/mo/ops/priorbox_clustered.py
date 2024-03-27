# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.layout import get_width_dim, get_height_dim
from openvino.tools.mo.front.extractor import attr_getter
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class PriorBoxClusteredOp(Op):
    op = 'PriorBoxClustered'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.priorbox_clustered_infer,
            'type_infer': self.type_infer,
            'clip': True,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'width',
            'height',
            'flip',
            'clip',
            'variance',
            'img_size',
            'img_h',
            'img_w',
            'step',
            'step_h',
            'step_w',
            'offset'
        ]

    def backend_attrs(self):
        return [
            ('clip', lambda node: int(node.clip)),  # We need to convert this boolean attribute value to int to keep
            # forward compatibility with OV 2021.2
            'img_h',
            'img_w',
            'step',
            'step_h',
            'step_w',
            'offset',
            ('variance', lambda node: attr_getter(node, 'variance')),
            ('width', lambda node: attr_getter(node, 'width')),
            ('height', lambda node: attr_getter(node, 'height'))
        ]

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(np.float32)

    @staticmethod
    def priorbox_clustered_infer(node: Node):
        layout = node.graph.graph['layout']
        data_shape = node.in_node(0).shape
        num_ratios = len(node.width)

        if node.has_and_set('V10_infer'):
            assert node.in_node(0).value is not None
            node.out_port(0).data.set_shape([2, np.prod(node.in_node(0).value) * num_ratios * 4])
        else:
            res_prod = data_shape[get_height_dim(layout, 4)] * data_shape[get_width_dim(layout, 4)] * num_ratios * 4
            node.out_port(0).data.set_shape([1, 2, res_prod])
