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

from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.squeeze import Squeeze


class GatherTreeNormalizer(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='GatherTree'):
            name = node.soft_get('name', node.id)
            assert 3 in node.in_ports() and not node.in_port(3).disconnected()

            end_token_shape = node.in_port(3).data.get_shape()
            assert end_token_shape is not None
            if end_token_shape.size == 1 and end_token_shape.ndim == 1:
                squeeze = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                           {'name': name + '/Squeeze', 'override_output_shape': True})
                node.in_port(3).get_connection().insert_node(squeeze)