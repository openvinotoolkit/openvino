"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np

from extensions.front.AttributedClampNormalizer import AttributedClampNormalizer
from extensions.ops.activation_ops import HSigmoid
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, rename_nodes


def replace_with_hsigmoid(graph: Graph, match: [dict, SubgraphMatch]):
    add = match['add']
    mul_2 = match['mul_2']

    # determine the input port of Add and Mul which gets the 'input' node output
    add_input_port_idx = int(add.in_port(0).get_connection().get_source().node.soft_get('op') == 'Const')
    mul_2_name = mul_2.soft_get('name', mul_2.id)

    hsigmoid = HSigmoid(graph, {}).create_node()
    hsigmoid.in_port(0).connect(add.in_port(add_input_port_idx).get_source())
    mul_2.out_port(0).get_connection().set_source(hsigmoid.out_port(0))

    rename_nodes([(mul_2, mul_2_name + '/TBR'), (hsigmoid, mul_2_name)])


class HSigmoidWithClamp(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with ReLU6 (Clamp) defining the HSigmoid function:
    HSigmoid(x) = Relu6(x + 3) / 6.0.
    """
    enabled = True

    def run_after(self):
        return [AttributedClampNormalizer]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('add', dict(op='Add')),
                ('const_0', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 0.0, atol=1e-6))),
                ('const_3', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 3.0, atol=1e-6))),
                ('const_6', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 6.0, atol=1e-6))),
                ('const_1_6', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 1 / 6.0, atol=1e-6))),
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
        replace_with_hsigmoid(graph, match)


class HSigmoidWithMinMax(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Min/Max defining the HSigmoid function:
    HSigmoid(x) = Min(Max(x + 3, 0), 6) / 6.0.
    """
    enabled = True

    def run_after(self):
        return [AttributedClampNormalizer]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('add', dict(op='Add')),
                ('const_0', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 0.0, atol=1e-6))),
                ('const_3', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 3.0, atol=1e-6))),
                ('const_6', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 6.0, atol=1e-6))),
                ('const_1_6', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 1 / 6.0, atol=1e-6))),
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
        replace_with_hsigmoid(graph, match)
