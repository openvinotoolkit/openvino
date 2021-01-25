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

from extensions.front.PowerToEltwises import PowerToEltwises
from extensions.ops.mvn import MVN
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Node, Graph


class MVNUnrolled(FrontReplacementSubgraph):
    enabled = True

    def run_after(self):
        return [PowerToEltwises]

    def pattern(self):
        log.debug('Enabled MVN replacement')
        return dict(
            nodes=[
                ('mean', dict(kind='op', op='ReduceMean')),
                ('stop_grad', dict(kind='op', op='StopGradient')),
                ('sqdiff', dict(kind='op', op='SquaredDifference')),
                ('variance', dict(kind='op', op='ReduceMean')),
                ('add', dict(kind='op', op='Add')),
                ('pow', dict(kind='op', op='Pow')),
                ('sub', dict(kind='op', op='Sub')),
                ('truediv', dict(kind='op', op='Div')),
            ],
            edges=[
                ('mean', 'stop_grad', {'in': 0}),
                ('stop_grad', 'sqdiff', {'in': 1}),
                ('sqdiff', 'variance', {'in': 0}),
                ('mean', 'sub', {'in': 1}),
                ('variance', 'add'),
                ('add', 'pow', {'in': 0}),
                ('pow', 'truediv', {'in': 1}),
                ('sub', 'truediv', {'in': 0}),
            ])

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        mvn = MVN(graph, dict(
            name=match['truediv'].name + '/MVN_',
            eps_mode='outside_sqrt',
            normalize_variance=1
        ))
        mvn.attrs['old_infer'] = mvn.attrs['infer']
        mvn.attrs['infer'] = __class__.infer

        mean_reduction = match['mean'].in_node(1)
        variance_reduction = match['variance'].in_node(1)
        pow2 = match['pow'].in_node(1)
        eps = match['add'].in_node(0 if match['add'].in_node(0).id != match['variance'].id else 1)

        new_subgraph = mvn.create_node([match['mean'].in_node(0), mean_reduction, variance_reduction, pow2, eps])

        match['truediv'].replace_node(new_subgraph)

    @staticmethod
    def infer(node: Node):
        if not (node.in_node(1).has_valid('value') and node.in_node(2).has_valid('value')):
            log.warning('Reduction indices for mean and variance for MVN node {} are not constants'.format(node.name))
            return

        if not (all(node.in_node(1).value == node.in_node(2).value)):
            log.warning('Reduction indices for mean {} and variance {} do not match'.format(
                node.in_node(1).value,
                node.in_node(2).value
            ))
            return
        
        if not (node.in_node(3).has_valid('value') and node.in_node(4).has_valid('value')):
            log.warning('Power or/and epsilon values for MVN node {} are not constants'.format(node.name))
            return

        if node.in_node(3).value != 0.5:
            log.warning('Power for MVN node {} ({}) is not equal to 0.5'.format(node.name, node.in_node(3).value))
            return

        node['eps'] = node.in_node(4).value

        for i in range(2, 5):
            node.graph.remove_edge(node.in_node(i).id, node.id)
        node.old_infer(node)
        node['infer'] = node.old_infer
