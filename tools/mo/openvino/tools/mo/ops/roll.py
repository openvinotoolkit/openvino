# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.ops.op import Op


class Roll(Op):
    """
    Roll operation that shifts elements of a tensor along specified axes.
    """
    op = 'Roll'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset7',
            'infer': roll_infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
            'in_ports_count': 3,
            'out_ports_count': 1
        }, attrs)


class AttributedRoll(Op):
    """ Roll operation that shifts elements of a tensor along specified axes.
        This operation uses the same semantics as Roll but with shift and axes specified as attributes.
        Shift and axes are specified as attributes in MxNet.
    """

    op = 'AttributedRoll'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'infer': None,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'shift': None,
            'axes': None
        }, attrs)


def roll_infer(node: Node):
    PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'axis')
    copy_shape_infer(node)
