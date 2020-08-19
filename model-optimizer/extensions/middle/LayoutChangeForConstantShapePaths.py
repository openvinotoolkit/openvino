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
from collections import deque
from typing import List, Set

from extensions.middle.InsertLayoutPropagationTransposes import is_output_data_in_correct_layout, \
    InsertLayoutPropagationTranspose, mark_input_as_in_correct_layout, mark_output_as_in_correct_layout
from extensions.ops.gather import Gather
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.graph.port import Port
from mo.middle.replacement import MiddleReplacementPattern


class LayoutChangeForConstantShapePaths(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC',
                       lambda graph: not graph.graph['cmd_params'].static_shape]
    force_clean_up = True

    def run_after(self):
        return [InsertLayoutPropagationTranspose]

    def run_before(self):
        return []

    @staticmethod
    def get_next_in_ports(in_port: Port) -> Set[Port]:
        next_in_ports = set()
        for out_port in in_port.node.out_ports().values():
            next_in_ports.update(out_port.get_destinations())
        return next_in_ports

    def mark_node_as_in_correct_layout_by_in_port(self, in_port):
        next_in_ports = self.get_next_in_ports(in_port)
        in_port.__setattr__('input_permutation', None)
        mark_input_as_in_correct_layout(in_port.node, in_port.idx)
        for port in next_in_ports:
            mark_output_as_in_correct_layout(port.get_source().node, port.get_source().idx)

    def find_shape_subgraph_endpoints(self, out_ports: List[Port], visited: set = None,
                                      action: callable = None) -> Set[Port]:
        """
        Searches for input ports of data dependent operations starting from output ports passed to the function.
        Condition for data dependent operations is absence of node output value.

        :param out_ports: list of output ports to start search from
        :param visited: set of input ports that were visited to avoid visiting them more than once
        :param action: function to call on the each input port of shape sub-graph
        :return: set of input ports of data dependent operations
        """
        if visited is None:
            visited = set()

        deque_of_in_ports = deque()
        for out_port in out_ports:
            deque_of_in_ports.extend(out_port.get_destinations())

        end_points_in_ports = set()
        while len(deque_of_in_ports):
            in_port = deque_of_in_ports.popleft()
            if in_port in visited:
                continue
            next_in_ports = self.get_next_in_ports(in_port)
            if any([port.data.get_value() is None for port in next_in_ports]):
                end_points_in_ports.add(in_port)
            else:
                deque_of_in_ports.extend(next_in_ports)
                if action is not None:
                    action(in_port)
            visited.add(in_port)
        return end_points_in_ports

    def find_and_replace_pattern(self, graph: Graph):
        shape_ops = graph.get_op_nodes(op='ShapeOf')

        # 1. Inserting Gather to N*C format on constant shape paths
        for shape in shape_ops:
            source_port = shape.in_port(0).get_source()
            if is_output_data_in_correct_layout(source_port.node, source_port.idx):
                continue  # data is already in N*C format

            name = shape.soft_get('name', shape.id)
            rank = source_port.data.get_shape().size

            if rank in [4, 5]:
                index = int64_array([0, *list(range(2, rank)), 1])
            else:
                continue  # data is layout independent

            gather = create_op_with_const_inputs(graph, op=Gather, port_value_dict={1: index, 2: int64_array(0)},
                                                 op_attrs={'name': name + '/GatherNCHWtoNHWC'})
            shape.out_port(0).get_connection().insert_node(gather)

        # 2. Inserting Gather/Transpose to NC* format
        shape_sub_graph_end_points = self.find_shape_subgraph_endpoints(
            [shape.out_port(0) for shape in shape_ops], None, self.mark_node_as_in_correct_layout_by_in_port)
        for in_port in shape_sub_graph_end_points:
            name = in_port.node.soft_get('name', in_port.node.id)
            shape = in_port.data.get_shape()

            should_switch_layout = not any([is_output_data_in_correct_layout(port.node, port.idx)
                                            for port in in_port.node.out_ports().values() if not port.disconnected()])
            should_insert_gather = should_switch_layout and len(shape) == 1 and shape.item(0) in [4, 5]
            should_insert_transpose = should_switch_layout and len(shape) in [4, 5]

            if should_insert_gather:
                # we should turn input permutation off to perform it with the following gather insertion
                in_port.__setattr__('input_permutation', None)
                index = int64_array([0, shape.item(0) - 1, *list(range(1, shape.item(0) - 1))])
                gather = create_op_with_const_inputs(graph, op=Gather,
                                                     port_value_dict={1: index, 2: int64_array(0)},
                                                     op_attrs={'name': name + '/GatherNHWCtoNCHW'})
                in_port.get_connection().insert_node(gather)
            elif should_insert_transpose:
                # we should turn input permutation off to perform it with the following transpose insertion
                in_port.__setattr__('input_permutation', None)
                order = int64_array([0, len(shape) - 1, *list(range(1, len(shape) - 1))])
                transpose = create_op_with_const_inputs(graph, op=Transpose, port_value_dict={1: order},
                                                        op_attrs={'name': name + '/TransposeNHWCtoNCHW',
                                                                  'override_output_shape': True})
                mark_input_as_in_correct_layout(transpose, 0)
                mark_output_as_in_correct_layout(transpose, 0)
                in_port.get_connection().insert_node(transpose)
            else:
                continue  # data is layout independent
