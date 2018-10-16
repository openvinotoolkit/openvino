"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx
from mo.graph.graph import replace_node
from extensions.ops.prelu import PreluOp
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.middle.pattern_match import check_node_usages_out_of_match


class PReLU(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op')),
                   ('pos_relu', dict(kind='op', op='Relu')),
                   ('neg', dict(kind='op', op='Neg')),
                   ('neg_relu', dict(kind='op', op='Relu')),
                   ('neg_1', dict(kind='op', op='Neg')),
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
                ('mul', 'add'),
            ],
            node_attrs=['kind', 'op'],
            edge_attrs=[])

    def replace_sub_graph(self, graph: nx.MultiDiGraph, match: dict):
        consumers = [n for n in match if n not in ['mul', 'op', 'add'] and not check_node_usages_out_of_match(match, n)]
        if consumers:
            log.warning('PReLU pattern was detected. Non pattern consumers of nodes: \'{}\' were found. Won\'t replace'
                        ''.format(', '.join([match[n].id for n in consumers])))
            return
        gamma = match['mul'].in_node(0) if match['mul'].in_node(1).id == match['neg_1'].id else match['mul'].in_node(1)
        prelu_op = PreluOp(graph, {'name': '{}/PReLU'.format(match['add'].id)}).create_node([match['op'], gamma])
        if match['add'].has_valid('is_output') and match['add'].is_output:
            match['add'].is_output = False
            prelu_op['is_output'] = True
        replace_node(match['add'], prelu_op)
        log.debug('PReLU pattern starting from \'{}\' was collapsed to \'{}\''.format(match['op'].id, prelu_op.id))
