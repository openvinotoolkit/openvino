"""
 Copyright (c) 2017-2019 Intel Corporation

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

from extensions.front.squared_difference import SquaredDifference
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Node, Graph
from extensions.ops.elementwise import Mul, Add
from mo.ops.op import Op


class MVN(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [SquaredDifference]

    def pattern(self):
        log.debug('Enabled MVN replacement')
        return dict(
            nodes=[
                ('mean', dict(op='ReduceMean')),
                ('stop_grad', dict(op='StopGradient')),
                ('sqdiff', dict(op='SquaredDifference')),
                ('variance', dict(op='ReduceMean')),
                ('squeeze_mean', dict(op='Squeeze')),
                ('squeeze_variance', dict(op='Squeeze')),
                ('fbn', dict(op='FusedBatchNorm')),
            ],
            edges=[
                ('mean', 'stop_grad', {'in': 0}),
                ('stop_grad', 'sqdiff', {'in': 1}),
                ('sqdiff', 'variance', {'in': 0}),
                ('mean', 'squeeze_mean', {'in': 0}),
                ('variance', 'squeeze_variance', {'in': 0}),
                ('squeeze_mean', 'fbn', {'in': 3}),
                ('squeeze_variance', 'fbn', {'in': 4}),
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        fbn = match['fbn']
        input = fbn.in_node(0)
        log.debug('Found potential MVN pattern after {} with name {}'.format(input.op, input.name))
        if input.id != match['mean'].in_node(0).id or input.id != match['sqdiff'].in_node(0).id:
            return

        log.debug('Confirmed MVN pattern after {} with name {}'.format(input.op, input.name))
        MVN = Op.get_op_class_by_name('MVN')

        mvn = MVN(graph, dict(
            name=fbn.name + '/MVN_',
            eps=fbn.eps,
            required_reduction_indices=[1, 2] if fbn.data_format == b'NHWC' else [2, 3]
        ))
        mvn.attrs['old_infer'] = mvn.attrs['infer']
        mvn.attrs['infer'] = __class__.infer

        mul = Mul(graph, dict(operation='mul', name=fbn.name + '/Mul_'))
        add = Add(graph, dict(operation='sum', name=fbn.name + '/Add_'))

        input_gamma = fbn.in_node(1)
        input_beta = fbn.in_node(2)

        mean_reduction = match['mean'].in_node(1)
        variance_reduction = match['variance'].in_node(1)

        new_subgraph = add.create_node([
            mul.create_node([
                mvn.create_node([input, mean_reduction, variance_reduction]),
                input_gamma
            ]),
            input_beta
        ])
        fbn.replace_node(new_subgraph)

    @staticmethod
    def infer(node: Node):
        if not (node.in_node(1).has_valid('value') and node.in_node(2).has_valid('value')):
            log.warning('Reduction indices for mean and variance for MVN node {} are not constants'.format(node.name))
            return

        if not (all(node.in_node(1).value == node.required_reduction_indices) and
                    all(node.in_node(2).value == node.required_reduction_indices)):
            log.warning('Reduction indices for mean {} and variance {} do not match required ones {}'.format(
                node.in_node(1).value,
                node.in_node(2).value,
                node.required_reduction_indices
            ))
            return

        node.graph.remove_edge(node.in_node(2).id, node.id)
        node.graph.remove_edge(node.in_node(1).id, node.id)
        node.old_infer(node)
