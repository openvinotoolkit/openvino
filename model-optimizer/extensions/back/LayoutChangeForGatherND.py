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

from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Port
from mo.back.replacement import BackReplacementPattern


class LayoutChangeForGatherND(BackReplacementPattern):
    """
    Return original layout for inputs and output of GatherND operation
    since the operation is designed for NHWC layout.
    """
    enabled = True
    force_shape_inference = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']

    @staticmethod
    def insert_transpose(graph: Graph, input_port: Port, before_input=True):
        input_rank = len(input_port.data.get_shape())
        if input_rank > 3:
            if before_input:
                axis_order = np.concatenate((int64_array([0]),
                                             int64_array(list(range(2, input_rank))),
                                             int64_array([1])))
                source_node = input_port.get_source().node
                transpose_name = source_node.soft_get('name', source_node.id) + '/TransposeToNHWC'
            else:
                axis_order = np.concatenate(
                    (int64_array([0]),
                     int64_array([input_rank - 1]),
                     int64_array(list(range(1, input_rank - 1)))))
                transpose_name = input_port.node.soft_get('name', input_port.node.id) + '/TransposeToNCHW'
                input_port.node['need_shape_inference'] = True
                input_port.node['override_output_shape'] = True
            transpose = create_op_with_const_inputs(graph, Transpose, {1: axis_order}, {'name': transpose_name})
            input_port.get_connection().insert_node(transpose)
            transpose['need_shape_inference'] = True
            transpose['override_output_shape'] = True

    def find_and_replace_pattern(self, graph: Graph):
        for gathernd in graph.get_op_nodes(type='GatherND'):
            self.insert_transpose(graph, gathernd.in_port(0), before_input=True)
            self.insert_transpose(graph, gathernd.in_port(1), before_input=True)
            self.insert_transpose(graph, gathernd.out_port(0), before_input=False)
