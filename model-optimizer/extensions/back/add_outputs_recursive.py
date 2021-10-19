# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from extensions.ops.If import If
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


# Add Result (and Unsqueeze is add_unsqueeze=True) to node port port_num in graph cur_graph.
# New nodes will have internal id equal to cur_max_layer_id + 1 (and cur_max_layer_id + 2 if 2 nodes were added)
# New nodes will be inserted in tracks on position i.
def add_output_in_body(node, port_num, cur_graph, cur_max_layer_id, tracks, track_index, add_unsqueeze=True):
    port = node.out_port(port_num)
    if add_unsqueeze:
        unsq_name = port.node.soft_get('name', port.node.id) + "/Unsqueeze"
        unsq_node = create_op_node_with_second_input(cur_graph, Unsqueeze,
                                                     int64_array([0]),
                                                     {'name': unsq_name})
        port.connect(unsq_node.in_port(0))
        unsq_node['internal_layer_id'] = cur_max_layer_id + 1
        cur_max_layer_id += 1
        tracks.insert(track_index, {'node': unsq_node, 'graph': cur_graph})
        port = unsq_node.out_port(0)

    out_name = port.node.soft_get('name', port.node.id) + ":" + str(port_num)
    res_node = Result(cur_graph, {'name': out_name}).create_node()
    port.connect(res_node.in_port(0))
    res_node['internal_layer_id'] = cur_max_layer_id + 1
    cur_max_layer_id += 1
    tracks.insert(track_index, {'node': res_node, 'graph': cur_graph})

    return res_node


class AddOutputRecursive(BackReplacementPattern):
    """
    Add output to node inside loops. Path to node set in 'additional_outputs' attribute of graph.
    Path structure: [node_loop_1, loop_2_in_loop_1,.., if_node, [then_list, else_list]]
    After if operation should be sub-list with 2 elements then_list and else_list where each is one node or list of
    nodes in according path
    """
    enabled = False
    run_not_recursively = True

    @staticmethod
    def add_output_for_path(graphs_nodes_path):
        # add output to nodes according to path
        step_node = graphs_nodes_path[-1]['node']
        cur_graph = graphs_nodes_path[-1]['graph']

        ports_to_add_nodes = []
        for o_p in step_node.out_ports():
            ports_to_add_nodes.append(o_p)

        # update internal_layer_id for new Results
        for i in range(len(graphs_nodes_path)-1, 0, -1):
            cur_max_layer_id = max_internal_layer_id(cur_graph) + 1
            cur_loop_node = graphs_nodes_path[i-1]['node']
            new_out_ports = []
            if cur_loop_node.op is not 'If':
                # add Unsqueeze and Result for TensorIterator and Loop and update output_port_map
                for p_num in ports_to_add_nodes:
                    res_node = add_output_in_body(step_node, p_num, cur_graph, cur_max_layer_id,
                                                  graphs_nodes_path, i)

                    # IR reader fix output port map for Loop, but have not change for TensorIterator
                    new_port_id = len(cur_loop_node.out_ports())
                    if cur_loop_node.op == 'TensorIterator':
                        new_port_id = new_port_id + len(cur_loop_node.in_ports())
                    cur_loop_node.output_port_map.append({'axis': 0, 'stride': 1, 'part_size': 1, 'start': 0,
                                                          'end': -1, 'external_port_id': new_port_id,
                                                          'internal_layer_id': res_node['internal_layer_id']})
                    port_id = new_port_id
                    if cur_loop_node.op == 'TensorIterator':
                        port_id = port_id - len(cur_loop_node.in_ports())

                    new_out_ports.append(port_id)
                    cur_loop_node.add_output_port(port_id)
            else:
                # add Result nodes for If and update output_id
                for p_num in ports_to_add_nodes:
                    res_node = add_output_in_body(step_node, p_num, cur_graph, cur_max_layer_id, graphs_nodes_path, i,
                                                  add_unsqueeze=False)

                    if cur_loop_node.then_graph == cur_graph:
                        new_port_id = len(cur_loop_node.out_ports())
                        res_node['output_id'] = new_port_id
                        cur_loop_node.add_output_port(new_port_id)
                        new_out_ports.append(new_port_id)
                    else:
                        res_node['output_id'] = list(cur_loop_node.out_ports().keys())[-1]
            ports_to_add_nodes = new_out_ports
            step_node = cur_loop_node
            cur_graph = graphs_nodes_path[i-1]['graph']

        i = 0
        for p_num in ports_to_add_nodes:
            port = step_node.out_port(p_num)
            out_name = step_node.soft_get('name', step_node.id) + "." + str(p_num)
            res_node = Result(cur_graph, {'name': out_name}).create_node()
            port.connect(res_node.in_port(0))
            # add name of Result to fw_tensor_debug_info to avoid renaming
            if step_node.out_nodes()[p_num].has_and_set('fw_tensor_debug_info'):
                step_node.out_nodes()[p_num]['fw_tensor_debug_info'].append(out_name)
            else:
                step_node.out_nodes()[p_num]['fw_tensor_debug_info'] = [[out_name, out_name]]
            if step_node.op == 'TensorIterator':
                step_node.out_edges()[len(step_node.out_edges())-1]['external_port_id'] = p_num + \
                                                                                          len(step_node.in_ports())
            graphs_nodes_path.insert(0, {'node': res_node, 'graph': cur_graph})
            i += 1
        return graphs_nodes_path

    @staticmethod
    def infer_shapes_of_nodes_in_path(graphs_nodes_path):
        # update shape for new or updated nodes
        for i in range(len(graphs_nodes_path) - 1, -1, -1):
            step_node = graphs_nodes_path[i]['node']
            # update shapes for Loop, TI, If, Unsqueeze
            # Result to end node in path added to existing port with already calculated shapes
            for p_num in step_node.out_ports():
                if not step_node.out_port(p_num).disconnected():
                    if step_node.op == 'TensorIterator':
                        ti_infer(step_node, p_num)
                    elif step_node.op == 'Loop':
                        loop_infer(step_node, p_num)
                    elif step_node.op == 'Unsqueeze':
                        assert step_node.in_port(1).get_source().node.has('value')
                        axis = step_node.in_port(1).get_source().node.value[0]
                        out_shape = list(step_node.in_port(0).get_source().data.get_shape())
                        out_shape.insert(axis, 1)
                        step_node.out_port(p_num).data.set_shape(out_shape)
                    elif step_node.op == 'If':
                        If.update_if_output_ports_shape(step_node)

    @staticmethod
    def split_path_to_simple_tracks(graph, path):
        # Split complex path into simple linear tracks.
        # In path after If node list with 2 sub-lists should be. In this function such path is split into 2 tracks:
        # one for each sublist with linear structure.
        # Number of tracks got from path is 2 * number of If operations in path.
        # Track is looks like list of paths with 2 fields : list of nodes on current path and list of according graphs
        # Example:
        # input path : [loop_1, loop_2, if_1, [[loop3_1, node_1], [node_2]]]
        # output track: [{'nodes': [loop_1, loop_2, if_1, loop3_1, node_1],
        #                 'graphs':[graph, loop_1.body, loop_2.body, if.then_graph, loop3_1.body]},
        #                {'nodes': [loop_1, loop_2, if_1, node_2],
        #                 'graphs':[graph, loop_1.body, loop_2.body, if.else_graph]}]

        # structure to save tracks
        # list with tracks, each track is list of pairs {'node', 'graph'}
        paths_nodes_graphs = list()
        paths_nodes_graphs.append([])
        # stack for sub-graphs that will be traversed in future
        future_graphs_stack = [graph]
        # index for track that we currently fill
        track_idx = 0
        # save lists that were started but not finished during processing
        lists_stack = [{'list': path, 'pos': -1}]
        while len(lists_stack) != 0:
            cur_list_pos = lists_stack.pop(-1)
            # current list to process
            cur_list = cur_list_pos['list']
            # index in current list/sub-list
            list_idx = cur_list_pos['pos'] + 1
            while list_idx < len(cur_list):
                el = cur_list[list_idx]
                if isinstance(el, (list, np.ndarray)):
                    lists_stack.append({'list': cur_list, 'pos': list_idx})
                    # if we have previous node non-list then current sublist is for If node
                    # and new tracks should be added for sub-graphs (the first subgraph will continue current track)
                    if list_idx != 0 and isinstance(cur_list[list_idx - 1], str):
                        for i in range(len(el) - 1):
                            # copy all nodes from existing track to new one
                            paths_nodes_graphs.append(paths_nodes_graphs[-1][:])
                    # new sublist started, so reset index
                    cur_list = el
                    list_idx = 0
                else:
                    assert isinstance(el, str)
                    cur_graph = future_graphs_stack.pop(-1)
                    step_node = Node(cur_graph, el)
                    paths_nodes_graphs[track_idx].append({'node': step_node, 'graph': cur_graph})

                    # if node is not last, check that next node will be on current track or not
                    if list_idx != len(cur_list) - 1:
                        # so detect if we are in sublist with branches for If
                        # then in stack sublist is not the first node of list
                        # and have previous node with If operation name
                        if len(lists_stack) != 0 and lists_stack[-1]['pos'] != 0 and \
                                isinstance(lists_stack[-1]['list'][lists_stack[-1]['pos']-1], str):
                            # switch to next track
                            if list_idx != len(cur_list) - 1:
                                track_idx += 1
                        else:
                            assert step_node.has_and_set('sub_graphs'), "Node without sub-graphs is not last in path"
                            # the first graph should be first in traverse
                            for sub_graphs_name in reversed(step_node['sub_graphs']):
                                future_graphs_stack.append(step_node[sub_graphs_name])
                    list_idx += 1

        return paths_nodes_graphs

    def find_and_replace_pattern(self, graph: Graph):

        if 'additional_outputs' not in graph.graph:
            return

        path = graph.graph['additional_outputs']
        paths_nodes_graphs = self.split_path_to_simple_tracks(graph, path)

        paths_nodes_graphs_old = []
        for i in range(len(paths_nodes_graphs)):
            paths_nodes_graphs_old.append(paths_nodes_graphs[i][:])
            paths_nodes_graphs[i] = self.add_output_for_path(paths_nodes_graphs[i])

        for i in range(len(paths_nodes_graphs)):
            self.infer_shapes_of_nodes_in_path(paths_nodes_graphs[i])

        new_nodes = []
        for i in range(len(paths_nodes_graphs)):
            # new Result added to main graph should be on last place
            if paths_nodes_graphs_old[i][0]['node'] != paths_nodes_graphs[i][0]['node']:
                new_nodes.append(paths_nodes_graphs[i][0]['node'])

        return new_nodes
