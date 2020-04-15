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

import numpy as np

from mo.graph.graph import Graph
from mo.middle.passes.eliminate import merge_data_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.error import Error


class SparseReshapeMiddleReplacer(MiddleReplacementPattern):
    """
    Removes SparseReshape operation if the old shape and the output shape are the same.
    """
    enabled = True

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('sparse_reshape', dict(op='SparseReshape')),
            ],
            edges=[
            ])

    def replace_pattern(self, graph: Graph, match: dict):
        sparse_reshape = match['sparse_reshape']

        input_shape_value = sparse_reshape.in_port(1).data.get_value()
        output_shape_value = sparse_reshape.out_port(1).data.get_value()

        if input_shape_value is None or output_shape_value is None:
            raise Error("Input shape and output shape values must be defined for node {}".format(sparse_reshape.id))
        if not np.array_equal(input_shape_value, output_shape_value):
            raise Error("Input shape and output shape values must be equal for node {}".format(sparse_reshape.id))

        input_data_node1 = sparse_reshape.in_node(0)
        input_data_node2 = sparse_reshape.in_node(1)
        output_data_node1 = sparse_reshape.out_node(0)
        output_data_node2 = sparse_reshape.out_node(1)
        graph.remove_edge(input_data_node1.id, sparse_reshape.id)
        graph.remove_edge(sparse_reshape.id, output_data_node1.id)
        graph.remove_edge(input_data_node2.id, sparse_reshape.id)
        graph.remove_edge(sparse_reshape.id, output_data_node2.id)
        merge_data_nodes(graph, output_data_node1, input_data_node1)
        merge_data_nodes(graph, output_data_node2, input_data_node2)
        graph.remove_nodes_from([sparse_reshape.id, input_data_node1.id, input_data_node2.id])

        # TODO: investigate why this second way does not work
        #sparse_reshape.out_port(0).get_connection().set_source(sparse_reshape.in_port(0).get_source())
        #sparse_reshape.out_port(1).get_connection().set_source(sparse_reshape.in_port(1).get_source())
        #sparse_reshape.in_port(0).get_connection().set_destination(sparse_reshape.in_port(0)
        #sparse_reshape.in_port(2).disconnect()
        #graph.remove_nodes_from([sparse_reshape.id])
