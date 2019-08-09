"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.front.PowerToEltwises import PowerToEltwises
from extensions.front.sub import Sub
from extensions.ops.prelu import PreluOp
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.middle.pattern_match import check_node_usages_out_of_match


class PReLU(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [PowerToEltwises]

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op')),
                   ('pos_relu', dict(kind='op', op='ReLU')),
                   ('neg', dict(kind='op', op='Power', scale=-1, power=1, shift=0)),
                   ('neg_relu', dict(kind='op', op='ReLU')),
                   ('neg_1', dict(kind='op', op='Power', scale=-1, power=1, shift=0)),
                   ('mul', dict(kind='op', op='Mul')),
                   ('add', dict(kind='op', op='Add')),
                   ],
            edges=[
                ('op', 'pos_relu'),
                ('op', 'neg'),
                ('pos_relu', 'add'),
                ('neg', 'neg_relu'),
                ('neg_relu', 'neg_1'),
                ('neg_1', 'mul'),
                ('mul', 'add')
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        consumers = [n for n in match if n not in ['mul', 'op', 'add'] and not check_node_usages_out_of_match(match, n)]
        if consumers:
            log.warning('PReLU pattern was detected. Non pattern consumers of nodes: "{}" were found. Won\'t replace'
                        ''.format(', '.join([match[n].id for n in consumers])))
            return
        gamma = match['mul'].in_node(0) if match['mul'].in_node(1).id == match['neg_1'].id else match['mul'].in_node(1)
        prelu_node = PreluOp(graph, {'name': '{}/PReLU'.format(match['add'].id)}).create_node([match['op'], gamma])
        match['add'].replace_node(prelu_node)
        log.debug('PReLU pattern starting from "{}" was collapsed to "{}"'.format(match['op'].id, prelu_node.id))


class PReLUWithAbs(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [Sub]

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op')),
                   ('relu', dict(kind='op', op='ReLU')),
                   ('abs', dict(kind='op', op='Abs')),
                   ('sub', dict(kind='op', op='Sub')),
                   ('mul', dict(kind='op', op='Mul')),
                   ('mul_1', dict(kind='op', op='Mul')),
                   ('add', dict(kind='op', op='Add')),
                   ],
            edges=[
                ('op', 'relu'),
                ('op', 'abs'),
                ('op', 'sub'),
                ('abs', 'sub'),
                ('sub', 'mul'),
                ('mul', 'mul_1'),
                ('relu', 'add'),
                ('mul_1', 'add'),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        consumers = [n for n in match if
                     n not in ['mul', 'mul_1', 'op', 'add', 'abs', 'sub'] and not check_node_usages_out_of_match(match,
                                                                                                                 n)]
        if consumers:
            log.warning('PReLUWithAbs pattern was detected. Non pattern consumers of nodes: "{}" were found. Won\'t '
                        'replace '.format(', '.join([match[n].id for n in consumers])))
            return
        gamma = match['mul'].in_node(0) if match['mul'].in_node(1).id == match['sub'].id else match['mul'].in_node(1)
        prelu_node = PreluOp(graph, {'name': '{}/PReLU'.format(match['add'].id)}).create_node([match['op'], gamma])
        match['add'].replace_node(prelu_node)
        log.debug('PReLUWithAbs pattern starting from "{}" was collapsed to "{}"'.format(match['op'].id, prelu_node.id))
