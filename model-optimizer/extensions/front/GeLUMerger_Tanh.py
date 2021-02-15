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

from math import sqrt, fabs, pi

from extensions.ops.gelu import GeLUOP
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class GeLUMergerTanh(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        log.info('Enabled GeLU Merger for approximation with TanH')
        return dict(
            nodes=[
                ('pow',  dict(op='Pow')),
                ('mul',  dict(op='Mul')),
                ('mul0', dict(op='Mul')),
                ('mul1', dict(op='Mul')),
                ('mul2', dict(op='Mul')),
                ('tanh', dict(op='Tanh')),
                ('add',  dict(op='Add')),
                ('add0', dict(op='Add')),
                ('mul_param', dict(op='Const')),
                ('mul0_param', dict(op='Const')),
                ('mul1_param', dict(op='Const')),
            ],
            edges=[
                ('pow',  'mul'),
                ('mul',  'add'),
                ('add',  'mul0'),
                ('mul0', 'tanh'),
                ('tanh', 'add0'),
                ('add0', 'mul1'),
                ('mul1', 'mul2'),
                ('mul_param',  'mul'),
                ('mul0_param', 'mul0'),
                ('mul1_param', 'mul1'),
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        # Gaussian Error Linear Unit, TanH based approximation:
        # 0.5*x*(1 + tanh([sqrt(2/pi)]*[x + 0.044715x3])
        inp_port = match['pow'].in_port(0).get_source()
        inp = inp_port.node
        log.debug('Found potential TanH-based GeLU pattern after {} with name {}'.format(inp.op, inp.name))

        # take the values of the mul ops
        mul_param  = match['mul_param']
        mul0_param = match['mul0_param']
        mul1_param = match['mul1_param']
        if mul0_param.value.size == 1 and mul_param.value.size == 1 and mul1_param.value.size == 1:
            mul_param = match['mul_param'].value.item()
            mul0_param = match['mul0_param'].value.item()
            mul1_param = match['mul1_param'].value.item()
            sqrt2pi = sqrt(2.0/pi)
            # check that the values match the approximation
            if fabs(mul0_param - sqrt2pi) < 1e-06 and fabs(mul_param - 0.044715) < 1e-06 and mul1_param == 0.5:
                log.debug('Confirmed TanH-based GELU pattern after {} with name {}'.format(inp.op, inp.name))
                gelu = GeLUOP(graph, dict(name=inp.name + '/GELU_')).create_node()
                inp_port.connect(gelu.in_port(0))
                match['mul2'].out_port(0).get_connection().set_source(gelu.out_port(0))
