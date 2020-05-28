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

from extensions.back.Reshape0DToSqueeze import Reshape0DToSqueeze
from extensions.back.ScalarConstNormalize import ScalarNormalize
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, Node
from mo.ops.reshape import Reshape
from mo.ops.result import Result


class TopKNormalizer(BackReplacementPattern):
    """
    The transformation converts the second input to the TopK layer from 0D to 1D.

    Also the transformation adds the Result Op if there are no consumers of TopK outputs. However the Result for output
    with values is not added if the node has attribute 'remove_values_output' which is set to True for Caffe models
    where ArgMax does not have separate output with values.

    TODO this pass should be removed when IE supports 0D tensors.
    """
    enabled = True

    def run_after(self):
        return [ScalarNormalize]

    def run_before(self):
        return [Reshape0DToSqueeze]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('result', {'type': 'TopK'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['result']
        is_scalar = graph.graph['cmd_params'].generate_experimental_IR_V10

        reshape = create_op_node_with_second_input(graph, Reshape, int64_array([]) if is_scalar else int64_array([1]),
                                                   {'override_output_shape': True})
        node.in_port(1).get_connection().insert_node(reshape)

        TopKNormalizer.normalize_outputs(node, graph)

    @staticmethod
    def normalize_outputs(node: Node, graph: Graph = None):
        """
        This function adds missed outputs for TopK node.
        """
        if graph == None:
            graph = node.graph

        if node.out_port(0).disconnected():
            output = Result(graph, {'name': node.name + '/Result_port_0/',
                                    'remove_from_xml': node.has_and_set('remove_values_output')}).create_node()
            node.out_port(0).get_connection().set_destination(output.in_port(0))
        if node.out_port(1).disconnected():
            output = Result(graph, {'name': node.name + '/Result_port_1/'}).create_node()
            node.out_port(1).get_connection().set_destination(output.in_port(0))
