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

import numpy as np

from extensions.ops.elementwise import Mul, Add, Sub
from extensions.ops.mvn import MVN
from extensions.ops.range import Range
from extensions.ops.rank import Rank
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Node, Graph
from mo.ops.const import Const
from mo.ops.unsqueeze import Unsqueeze


class MVNReplacer(FrontReplacementSubgraph):
    enabled = True

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
                ('fbn', dict(op=lambda op: op in ['BatchNormInference'])),
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
        output = fbn.out_node(0)
        log.debug('Found potential MVN pattern after {} with name {}'.format(input.op, input.name))
        if input.id != match['mean'].in_node(0).id or input.id != match['sqdiff'].in_node(0).id:
            return

        log.debug('Confirmed MVN pattern after {} with name {}'.format(input.op, input.name))

        mvn = MVN(graph, dict(
            name=fbn.name + '/MVN_',
            eps=fbn.eps,
            required_reduction_indices=[1, 2] if fbn.data_format == b'NHWC' else [2, 3]
        ))
        mvn.attrs['old_infer'] = mvn.attrs['infer']
        mvn.attrs['infer'] = __class__.infer
        mvn = mvn.create_node()
        mul = Mul(graph, dict(operation='mul', name=fbn.name + '/Mul_')).create_node()
        add = Add(graph, dict(operation='sum', name=fbn.name + '/Add_')).create_node()

        input_gamma = fbn.in_node(1)
        input_beta = fbn.in_node(2)

        mean_reduction = match['mean'].in_node(1)
        variance_reduction = match['variance'].in_node(1)
        # input.out_port(0).get_connection().add_destination(mvn.in_port(0))
        mvn.add_input_port(1)
        mvn.add_input_port(2)
        mean_reduction.out_port(0).get_connection().add_destination(mvn.in_port(1))
        variance_reduction.out_port(0).get_connection().add_destination(mvn.in_port(2))

        fbn.in_port(0).get_connection().set_destination(mvn.in_port(0))

        if fbn.data_format == b'NCHW':
            rank = Rank(graph, dict(name=fbn.name + '/rank_')).create_node()
            rank_out_id = len(input.out_ports().values())
            input.add_output_port(rank_out_id)
            input.out_port(rank_out_id).get_connection().set_destination(rank.in_port(0))
            const_1 = Const(graph, dict(name=fbn.name+'/const_val_1_', value=np.int32(1))).create_node()

            limit = Sub(graph, dict(name=fbn.name+'/limit_')).create_node()
            limit.in_port(0).get_connection().set_source(rank.out_port(0))
            limit.in_port(1).get_connection().set_source(const_1.out_port(0))
            expand_dims_range = Range(graph, dict(name=fbn.name+'/expand_dims_range_')).create_node()
            const_1.out_port(0).get_connection().add_destination(expand_dims_range.in_port(0))
            const_1.out_port(0).get_connection().add_destination(expand_dims_range.in_port(2))
            limit.out_port(0).get_connection().set_destination(expand_dims_range.in_port(1))

            new_gamma = Unsqueeze(graph, dict(name=fbn.name+'/new_gamma')).create_node()
            new_gamma.in_port(0).get_connection().set_source(input_gamma.out_port(0))
            expand_dims_range.out_port(0).get_connection().add_destination(new_gamma.in_port(1))
            new_beta = Unsqueeze(graph, dict(name=fbn.name+'/new_beta')).create_node()
            new_beta.in_port(0).get_connection().set_source(input_beta.out_port(0))
            expand_dims_range.out_port(0).get_connection().add_destination(new_beta.in_port(1))

            input_gamma = new_gamma
            input_beta = new_beta
        mul.in_port(0).get_connection().set_source(mvn.out_port(0))
        input_gamma.out_port(0).get_connection().add_destination(mul.in_port(1))

        new_subgraph = add
        new_subgraph.in_port(0).get_connection().set_source(mul.out_port(0))
        input_beta.out_port(0).get_connection().add_destination(new_subgraph.in_port(1))

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
