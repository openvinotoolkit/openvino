# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, Node
from mo.ops.result import Result
from mo.ops.unsqueeze import Unsqueeze
from mo.utils.error import Error


class AddOutputRecursive(BackReplacementPattern):
    """
    """
    enabled = True
    run_not_recursively = True

    def find_and_replace_pattern(self, graph: Graph):
        """
        If graph have 'additional_outputs' attribute, read list of nodes and add it to real result
        List structure: [node_loop_1, loop_2_in_loop_1,..,node_in_last_loop]
        ]
        """
        if 'additional_outputs' not in graph.graph:
            return

        path = graph.graph['additional_outputs']
        cur_graph = graph
        last_node = False
        step_node = None
        path_nodes = []
        for step in path[0:-1]:
            step_node = Node(cur_graph, step)
            path_nodes.append(step_node)
            if step_node.has_and_set('body'):
                cur_graph = step_node['body']
            else:
                raise Error("Not last node in list of nodes is not Loop/If/TI")

        step_node = Node(cur_graph, path[-1])
        path_nodes.append(step_node)

        ports_to_add_nodes = []
        for o_p in step_node.out_ports():
            ports_to_add_nodes.append(o_p)

        # update internal_layer_id for new Results
        for i in range(len(path)-1, 0, -1):
            cur_max_layer_id = len(cur_graph.nodes) + 1  # fix by finding real maximal internal_layer_id
            cur_loop_node = path_nodes[i-1]
            new_out_ports = []
            for p_num in ports_to_add_nodes:
                port = step_node.out_port(p_num)
                unsq_name = port.node.soft_get('name', port.node.id) + "/Unsqueeze"
                unsq_node = create_op_node_with_second_input(cur_graph, Unsqueeze,
                                                             int64_array([0]),
                                                             {'name': unsq_name})
                port.connect(unsq_node.in_port(0))
                out_name = unsq_name + ":" + str(p_num)
                res_node = Result(cur_graph, {'name': out_name}).create_node()
                unsq_node.out_port(0).connect(res_node.in_port(0))
                unsq_node['internal_layer_id'] = cur_max_layer_id + 1
                res_node['internal_layer_id'] = cur_max_layer_id + 2
                cur_max_layer_id += 2
                # infer shapes for new nodes
                step_node.infer(step_node)
                unsq_node.infer(unsq_node)
                res_node.infer(res_node)
                new_port_id = len(cur_loop_node.in_ports()) + len(cur_loop_node.out_ports())
                cur_loop_node.output_port_map.append({'axis': 0, 'stride': 1, 'part_size': 1, 'start': 0,
                                                      'end': -1, 'external_port_id': new_port_id,
                                                      'internal_layer_id': res_node['internal_layer_id']})
                cur_loop_node.add_output_port(new_port_id)
                new_out_ports.append(new_port_id)
            ports_to_add_nodes = new_out_ports
            step_node = cur_loop_node

        for p_num in ports_to_add_nodes:
            port = step_node.out_port(p_num)
            out_name = port.node.soft_get('name', step_node.id) + ":" + str(p_num)
            res_node = Result(graph, {'name': out_name}).create_node()
            port.connect(res_node.in_port(0))
            res_node.infer(res_node)

        step_node.infer(step_node)
