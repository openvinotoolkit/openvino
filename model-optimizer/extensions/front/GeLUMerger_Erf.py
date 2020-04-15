"""
 Copyright (C) 2017-2020 Intel Corporation

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

import logging as log
from math import sqrt, fabs

from extensions.ops.gelu import GeLUOP
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class GeLUMergerErf(FrontReplacementSubgraph):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]

    def pattern(self):
        log.info('Enabled GeLU Merger replacement for approximation with Erf')
        return dict(
            nodes=[
                ('mul',  dict(op='Mul')),
                ('mul0', dict(op='Mul')),
                ('div',  dict(op='Div')),
                ('erf',  dict(op='Erf')),
                ('add',  dict(op='Add')),
            ],
            edges=[
                ('mul', 'mul0'),
                ('div', 'erf'),
                ('erf', 'add'),
                ('add', 'mul0')
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        # Gaussian Error Linear Unit
        # f(x) = 0.5 * x * (1 + erf(x / sqrt(2))
        add = match['add']
        mul = match['mul']
        div = match['div']
        inp_port = div.in_port(0).get_source()
        inp = inp_port.node
        log.debug('Found potential Erf-based GeLU pattern after {} with name {}'.format(inp.op, inp.name))

        # take the values of the mul, add and div
        div_param = None
        mul_param = None
        add_param = None
        if div.in_port(0).get_source().node.soft_get('type') == 'Const':
            div_param = div.in_port(0).get_source().node.value
        elif div.in_port(1).get_source().node.soft_get('type') == 'Const':
            div_param = div.in_port(1).get_source().node.value

        if mul.in_port(0).get_source().node.soft_get('type') == 'Const':
            mul_param = mul.in_port(0).get_source().node.value
        elif mul.in_port(1).get_source().node.soft_get('type') == 'Const':
            mul_param = mul.in_port(1).get_source().node.value

        if add.in_port(0).get_source().node.soft_get('type') == 'Const':
            add_param = add.in_port(0).get_source().node.value
        elif add.in_port(1).get_source().node.soft_get('type') == 'Const':
            add_param = add.in_port(1).get_source().node.value

        if mul_param is None or div_param is None or add_param is None:
            return

        sqrt2 = sqrt(2.0)
        # check that the values match the approximation
        if fabs(div_param - sqrt2) < 1e-06 and mul_param == 0.5 and add_param == 1.0:
            log.debug('Confirmed Erf-based GELU pattern after {} with name {}'.format(inp.op, inp.name))
            gelu = GeLUOP(graph, dict(name=inp.name + '/GELU_')).create_node()
            inp_port.connect(gelu.in_port(0))
            match['mul0'].out_port(0).get_connection().set_source(gelu.out_port(0))
