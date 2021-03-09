"""
 Copyright (C) 2018-2021 Intel Corporation

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

from extensions.middle.MarkSubgraphsWithCorrectLayout import MarkSubGraphsWithCorrectLayout
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class MarkShapeOfSubgraphDataType(BackReplacementPattern):
    """
    This replacer marks op nodes in ShapeOf subgraphs with 'in_shape_subgraph' bool attribute and
    data nodes of float32 constants with 'correct_data_type' attribute.
    So that float Consts and Cast_to float will be kept in FP32 even if cmd_param --data_type=FP16 was specified.

    This is needed to enable conversion to FP16 even if values in ShapeOf subgraphs exceed max(float16)
    or because of FP16 lower precession shape inference is incorrect on some nodes (e.g. if Interpolate in scales mode
    accepts values from ShapeOf subgraph).

    This transformation must be run after shape inference and after all transformations that insert/modify
    Cast nodes in ShapeOf subgraphs therefore it's placed at the end of the back phase.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].data_type == 'FP16']

    def run_after(self):
        from extensions.back.pass_separator import BackFinish
        return [BackFinish]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        condition = lambda node: any([out_port.data.get_value() is not None for out_port in node.out_ports().values()])

        start_nodes = []
        shapeof_nodes = set(graph.get_op_nodes(op='ShapeOf'))
        for node in shapeof_nodes:
            start_nodes.extend([n for n in MarkSubGraphsWithCorrectLayout.get_output_nodes(node) if condition(n)])

        nodes_in_shapeof_subgraph = MarkSubGraphsWithCorrectLayout.bfs(start_nodes, shapeof_nodes,
                                                                       condition=condition,
                                                                       direction="bidirectional")
        for node in nodes_in_shapeof_subgraph:
            node['in_shape_subgraph'] = True
            if node.op == 'Const' and node.value.dtype == np.float32:
                node.out_node(0)['correct_data_type'] = True
