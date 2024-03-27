# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class ConstantFill(Op):
    """ Constant blob generation by broadcasting specified value to a given shape.

        It is assumed that there is no equivalent of this op in IE,
        so it is usually relevant to constant folding.
    """
    op = 'ConstantFill'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'input_as_shape': 1,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': self.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'input_as_shape',
            'fill_value'
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 1
        assert node.fill_value is not None
        assert node.input_as_shape

        shape = node.in_port(0).data.get_value()
        assert shape is not None

        if is_fully_defined(shape):
            node.out_port(0).data.set_value(np.full(shape, node.fill_value, np.float32))
        else:
            node.out_port(0).data.set_shape(shape)
