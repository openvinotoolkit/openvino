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
from extensions.back.ScalarConstNormalize import ScalarNormalize
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.reshape import Reshape


class NonMaximumSuppressionNormalizer(BackReplacementPattern):
    """
    Converts NonMaximumSuppression layer to the V7 specification.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_deprecated_IR_V7]

    def run_before(self):
        return [ScalarNormalize]

    def find_and_replace_pattern(self, graph: Graph):
        for nms in graph.get_op_nodes(type='NonMaxSuppression'):
            mapping = {'corner': 0, 'center': 1}

            if graph.graph['cmd_params'].generate_deprecated_IR_V7:
                nms['center_point_box'] = mapping[nms.box_encoding]

                # change the 0D inputs to 1D with shape [1]
                for port_id in range(2, 5):
                    if port_id in nms.in_ports() and not nms.in_port(port_id).disconnected():
                        reshape_1d = create_op_node_with_second_input(graph, Reshape, int64_array([1]),
                                                                      {'name': nms.soft_get('name') +
                                                                               '/Reshape_1D'.format(port_id),
                                                                       'override_output_shape': True,
                                                                       })
                        nms.in_port(port_id).get_connection().insert_node(reshape_1d)
