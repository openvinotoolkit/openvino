"""
 Copyright (c) 2019 Intel Corporation

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
from extensions.ops.ReduceOps import ReduceSum
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.power import Power
from mo.ops.reshape import Reshape


class ReplacePNormNodePattern(MiddleReplacementPattern):
    """
    PNorm operation should be replaced by operations: Power(P) -> Reshape(n,c*g->n,g,c)-> ReduceSum(axis=1)-> Power(1/P)
    """
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(op='pnorm'))],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        shape = node.in_port(0).data.get_shape().copy()

        assert shape[1] % node.group == 0

        power_node = Power(graph, attrs={'name': node.id + '_power',
                                         'power': node.p}).create_node()

        reshape_node = create_op_node_with_second_input(graph, Reshape,
                                                        int64_array([shape[0], shape[1] / node.group, node.group]),
                                                        {'name': node.id + '_reshape'})
        reshape_node.in_port(0).connect(power_node.out_port(0))

        reducesum_node = create_op_node_with_second_input(graph, ReduceSum,
                                                          int64_array([2]),
                                                          {'name': node.id + '_sum', 'keep_dims': False})
        reducesum_node.in_port(0).connect(reshape_node.out_port(0))

        invpower_node = Power(graph, attrs={'name': node.id + '_invpower',
                                            'power': 1.0 / node.p}).create_node()
        invpower_node.in_port(0).connect(reducesum_node.out_port(0))

        node.in_port(0).get_connection().set_destination(power_node.in_port(0))
        node.out_port(0).get_connection().set_source(invpower_node.out_port(0))
