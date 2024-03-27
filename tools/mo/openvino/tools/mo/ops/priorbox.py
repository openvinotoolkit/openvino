# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.layout import get_width_dim, get_height_dim
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import attr_getter, bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class PriorBoxOp(Op):
    op = 'PriorBox'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'flip': True,
            'clip': True,
            'scale_all_sizes': True,
            'max_size': mo_array([]),
            'min_size': mo_array([]),
            'aspect_ratio': mo_array([]),
            'density': mo_array([]),
            'fixed_size': mo_array([]),
            'fixed_ratio': mo_array([]),
            'in_ports_count': 2,
            'out_ports_count': 1,
            'type_infer': self.type_infer,
            'infer': self.priorbox_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'min_size',
            'max_size',
            'aspect_ratio',
            'flip',
            'clip',
            'variance',
            'img_size',
            'img_h',
            'img_w',
            'step',
            'step_h',
            'step_w',
            'offset',
            'density',
            'fixed_size',
            'fixed_ratio',
        ]

    def backend_attrs(self):
        return [
            ('flip', lambda node: int(node.flip)),  # We need to convert this boolean attribute value to int to keep
            # forward compatibility with OV 2021.2
            ('clip', lambda node: int(node.clip)),  # We need to convert this boolean attribute value to int to keep
            # forward compatibility with OV 2021.2
            'step',
            'offset',
            ('scale_all_sizes', lambda node: bool_to_str(node, 'scale_all_sizes')),
            ('min_size', lambda node: attr_getter(node, 'min_size')),
            ('max_size', lambda node: attr_getter(node, 'max_size')),
            ('aspect_ratio', lambda node: attr_getter(node, 'aspect_ratio')),
            ('variance', lambda node: attr_getter(node, 'variance')),
            ('density', lambda node: attr_getter(node, 'density')),
            ('fixed_size', lambda node: attr_getter(node, 'fixed_size')),
            ('fixed_ratio', lambda node: attr_getter(node, 'fixed_ratio')),
        ]

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(np.float32)

    @staticmethod
    def priorbox_infer(node: Node):
        layout = node.graph.graph['layout']
        data_shape = node.in_node(0).shape

        # calculate all different aspect_ratios (the first one is always 1)
        # in aspect_ratio 1/x values will be added for all except 1 if flip is True
        ar_seen = [1.0]
        ar_seen.extend(node.aspect_ratio.copy())
        if node.flip:
            for s in node.aspect_ratio:
                ar_seen.append(1.0 / s)

        ar_seen = np.unique(mo_array(ar_seen).round(decimals=6))

        num_ratios = 0
        if len(node.min_size) > 0:
            num_ratios = len(ar_seen) * len(node.min_size)

        if node.has_valid('fixed_size') and len(node.fixed_size) > 0:
            num_ratios = len(ar_seen) * len(node.fixed_size)

        if node.has_valid('density') and len(node.density) > 0:
            for d in node.density:
                if node.has_valid('fixed_ratio') and len(node.fixed_ratio) > 0:
                    num_ratios = num_ratios + len(node.fixed_ratio) * (pow(d, 2) - 1)
                else:
                    num_ratios = num_ratios + len(ar_seen) * (pow(d, 2) - 1)

        num_ratios = num_ratios + len(node.max_size)

        if node.has_and_set('V10_infer'):
            assert node.in_node(0).value is not None
            node.out_port(0).data.set_shape([2, np.prod(node.in_node(0).value) * num_ratios * 4])
        else:
            res_prod = data_shape[get_height_dim(layout, 4)] * data_shape[get_width_dim(layout, 4)] * num_ratios * 4
            node.out_port(0).data.set_shape([1, 2, res_prod])
