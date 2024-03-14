# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class DequantizeLinear(Op):
    op = 'DequantizeLinear'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'axis': None,
            'version': None,
            'infer': copy_shape_infer,
            'out_ports_count': 1,
            'in_ports_count': 3,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return ['axis']
