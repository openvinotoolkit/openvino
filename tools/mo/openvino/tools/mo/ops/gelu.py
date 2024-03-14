# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class GeLUOP(Op):
    op = 'Gelu'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'version': 'opset7',
            'infer': copy_shape_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        if self.get_opset() == 'opset7':
            return ['approximation_mode']
        else:
            return []
