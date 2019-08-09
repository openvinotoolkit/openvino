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
import numpy as np

from extensions.back.InsertLayoutPropagationTransposes import is_output_data_in_correct_layout
from extensions.ops.gather import Gather
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.ops.const import Const


class LayoutChangeForConstantShapePaths(BackReplacementPattern):
    enabled = False
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf',
                       lambda graph: graph.graph['cmd_params'].keep_shape_ops]
    force_clean_up = True

    @staticmethod
    def if_has_value(graph: Graph, node_name: str):
        return Node(graph, node_name).has_valid('value')

    def search_of_constant_path_end(self, graph: Graph, node_name: str, visited: set):
        from collections import deque
        d = deque()
        d.appendleft(node_name)
        ends = set()
        while len(d) != 0:
            cur_node = d.popleft()
            node = Node(graph, cur_node)
            if node.has_valid('permute_attrs'):
                node['permute_attrs'] = None
            for _, out_node_name in graph.out_edges(cur_node):
                if out_node_name not in visited:
                    if self.if_has_value(graph, out_node_name):
                        visited.add(cur_node)
                        d.extend([op for _, op in graph.out_edges(out_node_name)])
                    else:
                        ends.add(cur_node)
        return ends

    def find_and_replace_pattern(self, graph: Graph):
        # 1. Inserting Gather to N*C format on constant shape paths
        #   - Search for Shape ops
        #   - Inserting Gather after them in case of [4] or [5] output shape

        shape_ops = graph.get_op_nodes(op='ShapeOf')
        constant_shape_paths = set()
        gather_inserted = []

        for shape in shape_ops:
            output_port = shape.in_port(0).get_source()
            if is_output_data_in_correct_layout(output_port.node, output_port.idx):
                continue
            shape_of_shape_op_output = shape.out_node().shape

            if np.array_equal(shape_of_shape_op_output, [4]):
                index = np.array([0, 2, 3, 1])
            elif np.array_equal(shape_of_shape_op_output, [5]):
                index = np.array([0, 2, 3, 4, 1])
            else:
                continue

            const = Const(graph, {'value': index}).create_node()
            gather = Gather(graph, {'name': shape.name + '/GatherNCHWtoNHWC'}).create_node()

            shape.out_port(0).get_connection().set_source(gather.out_port(0))
            shape.out_port(0).connect(gather.in_port(0))
            const.out_port(0).connect(gather.in_port(1))

            constant_shape_paths.add(gather.id)
            gather_inserted.append(gather.id)

        # 2. Inserting Gather to NC* format
        #   - Search from Shape ops found in previous step for nodes without value that are n-th children of Shape op
        #       * MO can not propagate value, there is data path
        #   - Inserting Gather on ports which comes from operations in `constant_shape_paths` list

        constant_shape_ends = []

        for shape in shape_ops:
            constant_shape_ends.extend(self.search_of_constant_path_end(graph, node_name=shape.id,
                                                                        visited=constant_shape_paths))

        for end in constant_shape_ends:
            node = Node(graph, end)
            in_ports = [in_port for in_port in node.in_ports().values()
                        if in_port.get_source().node.id in constant_shape_paths]

            for in_port in in_ports:
                shape = in_port.data.get_shape()

                if np.array_equal(shape, [4]):
                    index = np.array([0, 3, 1, 2])
                elif np.array_equal(shape, [5]):
                    index = np.array([0, 2, 3, 4, 1])
                else:
                    continue

                const = Const(graph, {'value': np.array(index)}).create_node()
                gather = Gather(graph, {'name': node.name + '/GatherNHWCtoNCHW'}).create_node()

                in_port.get_source().connect(gather.in_port(0))
                in_port.get_connection().set_source(gather.out_port(0))
                const.out_port(0).connect(gather.in_port(1))
