# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.AttributedClampNormalizer import AttributedClampNormalizer
from openvino.tools.mo.ops.activation_ops import HSigmoid
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.middle.pattern_match import check_value
from openvino.tools.mo.utils.graph import Node


def replace_with_hsigmoid(graph: Graph, first_node: Node, last_node: Node):
    # determine the input port of first and last nodes which gets the 'input' node output
    add_input_port_idx = int(first_node.in_port(0).get_connection().get_source().node.soft_get('op') == 'Const')
    last_node_name = last_node.soft_get('name', last_node.id)

    hsigmoid = HSigmoid(graph, {}).create_node()
    hsigmoid.in_port(0).connect(first_node.in_port(add_input_port_idx).get_source())
    last_node.out_port(0).get_connection().set_source(hsigmoid.out_port(0))

    rename_nodes([(last_node, last_node_name + '/TBR'), (hsigmoid, last_node_name)])


class HSigmoidWithClamp(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with ReLU6 (Clamp) defining the HSigmoid function:
    HSigmoid(x) = Relu6(x + 3.0) / 6.0.
    """
    enabled = True

    def run_after(self):
        return [AttributedClampNormalizer]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('add', dict(op='Add')),
                ('const_0', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 0.0, atol=1e-6)))),
                ('const_3', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 3.0, atol=1e-6)))),
                ('const_6', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 6.0, atol=1e-6)))),
                ('const_1_6',
                 dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 1.0 / 6.0, atol=1e-6)))),
                ('clamp', dict(op='Clamp')),
                ('mul_2', dict(op='Mul')),
            ],
            edges=[
                ('input', 'add', {}),
                ('const_3', 'add', {}),
                ('add', 'clamp', {'in': 0}),
                ('const_0', 'clamp', {'in': 1}),
                ('const_6', 'clamp', {'in': 2}),
                ('clamp', 'mul_2', {}),
                ('const_1_6', 'mul_2', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        replace_with_hsigmoid(graph, match['add'], match['mul_2'])


class HSigmoidWithMinMax(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Min/Max defining the HSigmoid function:
    HSigmoid(x) = Min(Max(x + 3.0, 0), 6.0) / 6.0.
    """
    enabled = True

    def run_after(self):
        return [AttributedClampNormalizer]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('add', dict(op='Add')),
                ('const_0', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 0.0, atol=1e-6)))),
                ('const_3', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 3.0, atol=1e-6)))),
                ('const_6', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 6.0, atol=1e-6)))),
                ('const_1_6',
                 dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 1.0 / 6.0, atol=1e-6)))),
                ('max', dict(op='Maximum')),
                ('min', dict(op='Minimum')),
                ('mul_2', dict(op='Mul')),
            ],
            edges=[
                ('input', 'add', {'out': 0}),
                ('const_3', 'add', {}),
                ('add', 'max', {}),
                ('const_0', 'max', {}),
                ('max', 'min', {}),
                ('const_6', 'min', {}),
                ('min', 'mul_2', {}),
                ('const_1_6', 'mul_2', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        replace_with_hsigmoid(graph, match['add'], match['mul_2'])


class HSigmoidWithReluDiv(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Relu/Div defining the HSigmoid function:
    HSigmoid(x) = Min(Relu(x + 3.0), 6.0) / 6.0
    """
    enabled = True

    def run_after(self):
        return [AttributedClampNormalizer]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('add_const',
                 dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 3.0, atol=1e-6)))),
                ('add', dict(op='Add')),
                ('relu', dict(op='ReLU')),
                ('min_const',
                 dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 6.0, atol=1e-6)))),
                ('min', dict(op='Minimum')),
                ('div_const',
                 dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 6.0, atol=1e-6)))),
                ('div', dict(op='Div')),
            ],
            edges=[
                ('input', 'add', {'out': 0}),
                ('add_const', 'add', {}),
                ('add', 'relu', {}),
                ('relu', 'min', {}),
                ('min_const', 'min', {}),
                ('min', 'div', {}),
                ('div_const', 'div', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        replace_with_hsigmoid(graph, match['add'], match['div'])


class HSigmoidWithReluMul(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Relu/Mul defining the HSigmoid function:
    HSigmoid(x) = Min(Relu(x + 3.0), 6.0) * 1.0/6.0
    """
    enabled = True

    def run_after(self):
        return [AttributedClampNormalizer]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('add_const',
                 dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 3.0, atol=1e-6)))),
                ('add', dict(op='Add')),
                ('relu', dict(op='ReLU')),
                ('min_const',
                 dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 6.0, atol=1e-6)))),
                ('min', dict(op='Minimum')),
                ('mul_const',
                 dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 1.0 / 6.0, atol=1e-6)))),
                ('mul', dict(op='Mul')),
            ],
            edges=[
                ('input', 'add', {'out': 0}),
                ('add_const', 'add', {}),
                ('add', 'relu', {}),
                ('relu', 'min', {}),
                ('min_const', 'min', {}),
                ('min', 'mul', {}),
                ('mul_const', 'mul', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        replace_with_hsigmoid(graph, match['add'], match['mul'])
