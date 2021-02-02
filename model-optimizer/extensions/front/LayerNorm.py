"""
 Copyright (C) 2017-2021 Intel Corporation

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

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from extensions.ops.mvn import MVN


class LayerNorm(FrontReplacementSubgraph):
    # Compose part of the LayerNorm pattern to the MVN
    enabled = True

    def pattern(self):
        log.info('Enabled LayerNorm pattern recognition')
        return dict(
            nodes=[
                ('pool0', dict(op='ReduceMean')),
                ('pool1', dict(op='ReduceMean')),
                ('pow', dict(op='Pow')),
                ('div', dict(op='Div')),
                ('sqrt', dict(op='Pow')),
                ('add', dict(op='Add')),
                ('sub', dict(op='Sub')),
                ('pool0_param', dict(op='Const')),
                ('pool1_param', dict(op='Const')),
                ('add_param', dict(op='Const')),
                ('pow_param', dict(op='Const')),
            ],
            edges=[
                ('pool0', 'sub'),
                ('sub', 'pow'),
                ('pow', 'pool1'),
                ('pool1', 'add'),
                ('add', 'sqrt'),
                ('sqrt', 'div'),
                ('sub', 'div'),
                ('pool0_param', 'pool0'),
                ('pool1_param', 'pool1'),
                ('pow_param', 'sqrt'),
                ('add_param', 'add'),
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        inp = match['pool0']
        inp_port = inp.in_port(0).get_source()

        # take/check the values of the add, pow and axes for ReduceMean
        pow_param = match['pow_param']
        add_param = match['add_param']
        if add_param.value.size == 1 and pow_param.value.size == 1 and add_param.value.item() <= 1e-05 \
                and pow_param.value.item() == 0.5 and match['pool0_param'].value == match['pool1_param'].value:
            log.debug('Found LayerNorm pattern after {} with name {}'.format(inp_port.node.op, inp_port.node.name))
            mvn = create_op_with_const_inputs(graph, MVN, {1: match['pool1_param'].value},
                                              {'eps': add_param.value.item(), 'normalize_variance': 1,
                                               'eps_mode': 'inside_sqrt'})
            div_name = match['div'].soft_get('name', match['div'].id)
            rename_nodes([(match['div'], div_name + '/to_be_removed'), (mvn, div_name)])

            inp_port.connect(mvn.in_port(0))
            match['div'].out_port(0).get_connection().set_source(mvn.out_port(0))
