# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class ConvertLike(Op):
    op = 'ConvertLike'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'infer': copy_shape_infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
            'type_infer': self.type_infer,

            'in_ports_count': 2,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def type_infer(node: Node):
        assert node.is_in_port_connected(1), 'The second input is not connected for a node {}.' \
                                             ''.format(node.soft_get('name'), node.id)
        node.out_port(0).set_data_type(node.in_port(1).get_data_type())
