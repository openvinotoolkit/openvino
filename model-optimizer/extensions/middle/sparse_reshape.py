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

        # establish output shape if value of new shape is given as input
        new_shape_value = sparse_reshape.in_port(2).data.get_value()
        if output_shape_value is None and new_shape_value is not None:
            output_shape_value = new_shape_value
            if np.count_nonzero(output_shape_value == -1) == 1:
                elem = np.prod(input_shape_value) // np.prod(new_shape_value[new_shape_value != -1])
                output_shape_value[output_shape_value == -1] = elem

        if input_shape_value is None or output_shape_value is None:
            raise Error("Input shape and output shape values must be defined for node {}".format(sparse_reshape.id))
        if not np.array_equal(input_shape_value, output_shape_value):
            raise Error("Input shape and output shape values must be equal for node {}".format(sparse_reshape.id))

        nodes_to_remove = [sparse_reshape.id]
        if sparse_reshape.is_out_port_connected(0):
            sparse_reshape.out_port(0).get_connection().set_source(sparse_reshape.in_port(0).get_source())
            output_data_node = sparse_reshape.out_node(0)
            nodes_to_remove.append(output_data_node.id)
        else:
            input_data_node = sparse_reshape.in_node(0)
            nodes_to_remove.append(input_data_node.id)

        if sparse_reshape.is_out_port_connected(1):
            sparse_reshape.out_port(1).get_connection().set_source(sparse_reshape.in_port(1).get_source())
            output_data_node = sparse_reshape.out_node(1)
            nodes_to_remove.append(output_data_node.id)
        else:
            input_data_node = sparse_reshape.in_node(1)
            nodes_to_remove.append(input_data_node.id)

        graph.remove_nodes_from(nodes_to_remove)
