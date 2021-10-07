# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from extensions.ops.loop import Loop
from extensions.ops.tensor_iterator import TensorIterator
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, Node
from mo.ops.result import Result
from mo.ops.unsqueeze import Unsqueeze


def ti_find_iterations_count_for_output(step_node, output_rec):
    def check_axis_end(record):
        return 'axis' in record and record['axis'] is not None and \
            'end' in record and record['end'] is not None
    iterations_count = 1
    if not check_axis_end(output_rec):
        # in this case we dont concat outputs, so iterations count is not needed really
        return iterations_count

    if output_rec['end'] != -1:
        # get iterations count from output record
        iterations_count = (output_rec['end'] - output_rec['start']) / output_rec['stride']
        return iterations_count

    # find out iterations count from inputs
    for in_rec in step_node.input_port_map:
        if not check_axis_end(in_rec):
            continue
        if in_rec['end'] != -1:
            in_shape_axis = in_rec['end']
        else:
            in_shape_axis = step_node.in_port(in_rec['external_port_id']).data.get_shape()[in_rec['axis']]
        iterations_count = (in_shape_axis - in_rec['start']) / in_rec['stride']
        return iterations_count

    return iterations_count


# shape inference for TensorIterator
# copy shapes from internal nodes + insert correct iterations count where needed
def ti_infer(step_node, port_num):
    out_port_map = step_node.output_port_map
    int_layer_id = None
    port_num = port_num + len(step_node.in_ports())
    # find out which internal layer maps to port_num
    found_rec = None
    for om in out_port_map:
        if om['external_port_id'] == port_num:
            found_rec = om
            break
    assert found_rec is not None, \
        "External port {} is not connected with body in node {}".format(port_num,
                                                                        step_node.soft_get('name', step_node.id))

    # find out iterations count for TensorIterator to set output shape correctly
    iterations_count = ti_find_iterations_count_for_output(step_node, found_rec)

    int_node_name = TensorIterator.find_internal_layer_id(step_node.body, found_rec['internal_layer_id'])
    int_node = Node(step_node.body, int_node_name)
    assert int_node.op == 'Result'
    out_shape = int_node.in_port(0).data.get_shape().copy()
    port_num = port_num - len(step_node.in_ports())

    out_shape[0] = iterations_count

    step_node.out_port(port_num).data.set_shape(out_shape)


# shape inference for Loop
# copy shapes from internal nodes + insert correct iterations count where needed
def loop_infer(step_node, port_num):
    out_port_map = step_node.output_port_map
    int_layer_id = None
    iterations_count = Loop.iterations_count(step_node)
    for om in out_port_map:
        if om['external_port_id'] == port_num:
            int_layer_id = om['internal_layer_id']

    # used method from TensorIterator to find out node with needed internal layer id
    int_node_name = TensorIterator.find_internal_layer_id(step_node.body, int_layer_id)
    int_node = Node(step_node.body, int_node_name)
    assert int_node.op == 'Result'
    out_shape = int_node.in_port(0).data.get_shape().copy()

    out_shape[0] = iterations_count
    step_node.out_port(port_num).data.set_shape(out_shape)


def max_internal_layer_id(graph):
    max_int_layer_id = 0
    for n in graph.get_op_nodes():
        if n.has_and_set('internal_layer_id'):
            if n.internal_layer_id > max_int_layer_id:
                max_int_layer_id = n.internal_layer_id
    return max_int_layer_id


class AddOutputRecursive(BackReplacementPattern):
    """
    Add output to node inside loops. Path to node set in 'additional_outputs' attribute of graph.
    Path structure: [node_loop_1, loop_2_in_loop_1,.., if_node, [then_list, else_list]]
    After if operation should be sub-list with 2 elements then_list and else_list where each is one node or list of
    nodes in according path
    Created Result for path [loop1, loop2, node] will be loop1.loop2.node:<port number>
    """
    enabled = False
    run_not_recursively = True

    @staticmethod
    def add_output_for_path(nodes_path, graphs_path):
        # add output to nodes according to path
        step_node = nodes_path[-1]
        cur_graph = graphs_path[-1]

        final_res_name = ""
        for n in nodes_path:
            final_res_name += n.soft_get('name', n.id) + "_"
        final_res_name = final_res_name[:-1]

        ports_to_add_nodes = []
        for o_p in step_node.out_ports():
            ports_to_add_nodes.append(o_p)

        # update internal_layer_id for new Results
        for i in range(len(nodes_path)-1, 0, -1):
            cur_max_layer_id = max_internal_layer_id(cur_graph) + 1
            cur_loop_node = nodes_path[i-1]
            new_out_ports = []
            if cur_loop_node.op is not 'If':
                # add Unsqueeze and Result for TensorIterator and Loop and update output_port_map
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
                    nodes_path.insert(i, res_node)
                    nodes_path.insert(i, unsq_node)
                    graphs_path.insert(i, cur_graph)
                    graphs_path.insert(i, cur_graph)
                    # IR reader fix output port map for Loop, but have not change for TensorIterator
                    if cur_loop_node.op == 'TensorIterator':
                        new_port_id = len(cur_loop_node.out_ports()) + len(cur_loop_node.in_ports())
                    else:
                        new_port_id = len(cur_loop_node.out_ports())
                    cur_loop_node.output_port_map.append({'axis': 0, 'stride': 1, 'part_size': 1, 'start': 0,
                                                          'end': -1, 'external_port_id': new_port_id,
                                                          'internal_layer_id': res_node['internal_layer_id']})
                    if cur_loop_node.op == 'TensorIterator':
                        port_id = new_port_id - len(cur_loop_node.in_ports())
                    else:
                        port_id = new_port_id
                    new_out_ports.append(port_id)
                    cur_loop_node.add_output_port(port_id)
            else:
                # add Result nodes for If and update output_id
                for p_num in ports_to_add_nodes:
                    port = step_node.out_port(p_num)
                    out_name = step_node.soft_get('name', step_node.id) + ":" + str(p_num)
                    res_node = Result(cur_graph, {'name': out_name}).create_node()
                    port.connect(res_node.in_port(0))
                    res_node['internal_layer_id'] = cur_max_layer_id + 1
                    cur_max_layer_id += 1
                    nodes_path.insert(i, res_node)
                    graphs_path.insert(i, cur_graph)

                    if cur_loop_node.then_graph == cur_graph:
                        new_port_id = len(cur_loop_node.out_ports())
                        res_node['output_id'] = new_port_id
                        cur_loop_node.add_output_port(new_port_id)
                        new_out_ports.append(new_port_id)
                    else:
                        res_node['output_id'] = list(cur_loop_node.out_ports().keys())[-1]
            ports_to_add_nodes = new_out_ports
            step_node = cur_loop_node
            cur_graph = graphs_path[i-1]

        i = 0
        for p_num in ports_to_add_nodes:
            port = step_node.out_port(p_num)
            out_name = final_res_name + ":" + str(i)
            res_node = Result(cur_graph, {'name': out_name}).create_node()
            port.connect(res_node.in_port(0))
            # add name of Result to fw_tebsor_debug_info to avoid renaming
            if step_node.out_nodes()[p_num].has_and_set('fw_tensor_debug_info'):
                step_node.out_nodes()[p_num]['fw_tensor_debug_info'].append(out_name)
            else:
                step_node.out_nodes()[p_num]['fw_tensor_debug_info'] = [[out_name, out_name]]
            if step_node.op == 'TensorIterator':
                step_node.out_edges()[len(step_node.out_edges())-1]['external_port_id'] = p_num + \
                                                                                          len(step_node.in_ports())
            nodes_path.insert(len(nodes_path)-1, res_node)
            graphs_path.insert(len(nodes_path)-1, cur_graph)
            i += 1
        return nodes_path, graphs_path

    @staticmethod
    def infer_shapes_of_nodes_in_path(nodes_path):
        # update shape for new or updated nodes
        for i in range(len(nodes_path)-1, -1, -1):
            step_node = nodes_path[i]
            # infer shapes for new nodes
            for p_num in step_node.out_ports():
                if not step_node.out_port(p_num).disconnected():
                    if step_node.op == 'TensorIterator':
                        ti_infer(step_node, p_num)
                    elif step_node.op == 'Loop':
                        loop_infer(step_node, p_num)
            else:
                step_node.infer(step_node)

    @staticmethod
    def split_path_to_simple_tracks(graph, path):
        # Split complex path into simple linear tracks.
        # Track is looks like list of paths with 2 fields : list of nodes on current path and list of according graphs
        # Example:
        # input path : [loop_1, loop_2, if_1, [[loop3_1, node_1], [node_2]]]
        # output track: [{'nodes': [loop_1, loop_2, if_1, loop3_1, node_1],
        #                 'graphs':[graph, loop_1.body, loop_2.body, if.then_graph, loop3_1.body]},
        #                {'nodes': [loop_1, loop_2, if_1, node_2],
        #                 'graphs':[graph, loop_1.body, loop_2.body, if.else_graph]}]
        paths_nodes_graphs = []
        cur_graph = [graph]
        path_idx = 0
        graph_idx = 0
        i = 0
        lists_stack = []
        cur_list = path
        paths_nodes_graphs.append({'nodes': [], 'graphs': []})
        is_finished = False
        switch_path = False
        while not is_finished:
            while i < len(cur_list):
                el = cur_list[i]
                if isinstance(el, (list, np.ndarray)):
                    lists_stack.append({'list': cur_list, 'pos': i})
                    cur_list = el
                    if i != 0:
                        paths_nodes_graphs.append({'nodes': [], 'graphs': []})
                        paths_nodes_graphs[-1]['nodes'] = paths_nodes_graphs[-2]['nodes'].copy()
                        paths_nodes_graphs[-1]['graphs'] = paths_nodes_graphs[-2]['graphs'].copy()
                        switch_path = True
                    i = 0
                else:
                    assert isinstance(el, str)
                    step_node = Node(cur_graph[graph_idx], el)
                    paths_nodes_graphs[path_idx]['nodes'].append(step_node)
                    paths_nodes_graphs[path_idx]['graphs'].append(cur_graph[graph_idx])
                    if step_node.has_and_set('sub_graphs'):
                        for sub_graphs_name in step_node['sub_graphs']:
                            cur_graph.append(step_node[sub_graphs_name])
                    else:
                        if not switch_path:
                            assert i == len(cur_list) - 1, "Not last node in list of nodes is not Loop/If/TI"
                    if switch_path:
                        path_idx += 1
                        switch_path = False
                    i += 1
                    graph_idx += 1
            if len(lists_stack) != 0:
                cur_list_pos = lists_stack.pop(-1)
                cur_list = cur_list_pos['list']
                i = cur_list_pos['pos'] + 1
                path_idx -= 1
                graph_idx -= 1  # should check
            else:
                is_finished = True

        return paths_nodes_graphs

    def find_and_replace_pattern(self, graph: Graph):

        if 'additional_outputs' not in graph.graph:
            return

        path = graph.graph['additional_outputs']
        paths_nodes_graphs = self.split_path_to_simple_tracks(graph, path)

        for i in range(len(paths_nodes_graphs)):
            paths_nodes_graphs[i]['nodes'], paths_nodes_graphs[i]['graphs'] = self.add_output_for_path(
                paths_nodes_graphs[i]['nodes'], paths_nodes_graphs[i]['graphs'])

        for i in range(len(paths_nodes_graphs)):
            self.infer_shapes_of_nodes_in_path(paths_nodes_graphs[i]['nodes'])
