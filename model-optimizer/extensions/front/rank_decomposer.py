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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_nodes
from mo.ops.shape import Shape
from mo.ops.squeeze import Squeeze


class RankDecomposer(FrontReplacementOp):
    op = 'Rank'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        name = node.soft_get('name', node.id)

        assert node.has_valid('output_type'), \
            'Rank node should have `output_type` attribute, but it`s not for node {}'.format(name)

        shape_of = Shape(graph, {'name': name + '/shape_of', 'output_type': node.output_type}).create_node()
        rank_1d = Shape(graph, {'name': name + '/rank_of', 'output_type': node.output_type}).create_node()
        rank_0d = create_op_node_with_second_input(
            graph, Squeeze, int64_array(0), {'name': name + '/0d_rank_of'}, rank_1d)

        shape_of.out_port(0).connect(rank_1d.in_port(0))
        node.out_port(0).get_connection().set_source(rank_0d.out_port(0))
        node.in_port(0).get_connection().set_destination(shape_of.in_port(0))

        rename_nodes([(node, name + '/ToBeDeleted'), (rank_0d, name)])
