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
    InsertLayoutPropagationTranspose
from extensions.ops.gather import Gather
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.graph.port import Port
from mo.middle.replacement import MiddleReplacementPattern


class LayoutChangeForConstantShapePaths(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC',
                       lambda graph: graph.graph['cmd_params'].keep_shape_ops]
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

    def find_shape_subgraph_endpoints(self, out_ports: List[Port], visited: set = None) -> Set[Port]:
        """
        Searches for input ports of data dependent operations starting from output ports passed to the function.
        Condition for data dependent operations is absence of node output value.

        :param out_ports: list of output ports to start search from
        :param visited: set of input ports that were visited to avoid visiting them more than once
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

        # 2. Inserting Gather to NC* format
        shape_sub_graph_end_points = self.find_shape_subgraph_endpoints([shape.out_port(0) for shape in shape_ops])
        for in_port in shape_sub_graph_end_points:
            name = in_port.node.soft_get('name', in_port.node.id)
            rank = in_port.data.get_shape().item(0)

            should_insert_gather = rank in [4, 5] and not len(in_port.node.soft_get('correct_out_data_layout', {}))

            if should_insert_gather:
                # we should turn input permutation off to perform it with the following gather insertion
                in_port.__setattr__('input_permutation', None)
                index = int64_array([0, rank - 1, *list(range(1, rank - 1))])
            else:
                continue  # data is layout independent

            gather = create_op_with_const_inputs(graph, op=Gather, port_value_dict={1: index, 2: int64_array(0)},
                                                 op_attrs={'name': name + '/GatherNHWCtoNCHW'})
            in_port.get_connection().insert_node(gather)
