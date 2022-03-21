# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.middle.TensorIteratorLSTMToLSTMSequence import TensorIteratorLSTM

from openvino.tools.mo.middle.TensorIteratorBackEdge import BackEdgesMatching
from queue import Queue
from openvino.tools.mo.ops.identity import Identity
from openvino.tools.mo.ops.If import If
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.ops.result import Result
from openvino.tools.mo.front.tf.extractors.subgraph_utils import update_body_graph, convert_graph_inputs_to_parameters, \
    get_graph_proto, create_internal_graph

from openvino.tools.mo.ops.op import Op

from copy import deepcopy


class SwitchMergeToIf(FrontReplacementSubgraph):
    # def run_after(self):
    #     return [BackEdgesMatching, LoopConditionMatcher, DynamicDecoderConditionMatcher, ConditionChecks,
    #             SmartInputMatcher, SimpleInputMatcher, BackEdgeSimpleInputMatcher,
    #             SmartMatcherInputSlicingWithGather, TensorIteratorLSTM, TensorIteratorMerge,
    #             SmartOutputMatcher, SimpleOutputMatcher]
    @staticmethod
    def build_edges_dictionary(graph: Graph, from_dict, to_dict):
        edges = graph.edges
        for edge in edges:
            from_node_id = edge[0]
            to_node_id = edge[1]
            if not from_node_id in from_dict.keys():
                from_dict[from_node_id] = set()
            from_dict[from_node_id].add(to_node_id)

            if not to_node_id in to_dict.keys():
                to_dict[to_node_id] = set()
            to_dict[to_node_id].add(from_node_id)

    def find_and_replace_pattern(self, graph: Graph):
        from_node_dict = {}
        to_node_dict = {}

        SwitchMergeToIf.build_edges_dictionary(graph, from_node_dict, to_node_dict)
        for switch in graph.get_op_nodes(op='Switch'):
            if switch.has_and_set('deprecated'):
                continue
            SwitchMergeToIf.convert_to_if_by_switch(switch, from_node_dict, to_node_dict)

    @staticmethod
    def get_all_switches(condition_node, from_node_dict):
        graph = condition_node.graph
        switches = set()
        checking_nodes = Queue()
        skipping_nodes = set()
        for connected_id in from_node_dict[condition_node.soft_get('id', condition_node.soft_get('name'))]:
            checking_nodes.put(connected_id)

        while not checking_nodes.empty():
            node_id = checking_nodes.get()
            if node_id in skipping_nodes:
                continue
            else:
                skipping_nodes.add(node_id)

            node = Node(graph, node_id)
            if node.soft_get('op') == 'Switch':
                switches.add(node_id)
                continue
            for new_nodes_id in from_node_dict[node_id]:
                checking_nodes.put(new_nodes_id)
        return switches

    @staticmethod
    def check_on_tensor_iterator(node):
        types_of_ti_nodes =['Enter', 'Exit', 'NextIteration']
        if node.soft_get('op') in types_of_ti_nodes:
            return False
        return True

    @staticmethod
    def get_all_merges(graph, switches_set, from_node_dict, to_node_dict):
        checked_nodes = set()
        nodes_queue = Queue()
        merge_nodes = set()
        for switch_node_id in switches_set:
            for child_node in from_node_dict[switch_node_id]:
                nodes_queue.put(child_node)

        while not nodes_queue.empty():
            node_id = nodes_queue.get()
            node = Node(graph, node_id)
            op_node = node.soft_get('op')
            if node_id in checked_nodes:
                continue
            checked_nodes.add(node_id)
            if op_node == 'Switch':
                SwitchMergeToIf.convert_to_if_by_switch(node, from_node_dict, to_node_dict)
                continue
            if not SwitchMergeToIf.check_on_tensor_iterator(node):
                return set()
            if op_node == 'Merge':
                merge_nodes.add(node_id)
                continue
            if op_node == 'Result':
                continue
            for addition_nodes_id in from_node_dict[node_id]:
                nodes_queue.put(addition_nodes_id)
        return merge_nodes

    @staticmethod
    def convert_to_if(graph, condition_node_out_port, switches_set, merges_set):

        graph.stage = 'front'
        for switch_id in switches_set:
            merge_node = Node(graph, switch_id)
            for out_port in merge_node.out_ports(True).values():
                out_connection = out_port.get_connection()
                identity_name = switch_id + '/parameter_template'
                identity_node = Identity(graph, {'name': identity_name}).create_node()
                identity_node['parameter_template'] = True
                out_connection.set_source(identity_node.out_port(0))
                identity_node.in_port(0).get_connection().set_source(out_port)
        for merge_id in merges_set:
            merge_node = Node(graph, merge_id)
            for in_port in merge_node.in_ports(True).values():
                in_node = in_port.get_connection().get_source().node
                identity_name = merge_id + '/result_template'
                identity_node = Identity(graph, {'name': identity_name}).create_node()
                identity_node['result_template'] = True
                in_port.get_connection().add_destination(identity_node.in_port(0))
                in_port.get_connection().set_source(identity_node.out_port(0))
        params_info = {}
        for switch_node_id in switches_set:
            switch_node = Node(graph, switch_node_id)
            input_parameter_connection = switch_node.in_port(0).get_connection().get_source()
            while input_parameter_connection.node.soft_get('op') == 'Identity':
                input_parameter_connection = input_parameter_connection.node.in_port(
                    0).get_connection().get_source()
            else_branch_nodes = None
            then_branch_nodes = None
            if switch_node.is_out_port_connected(0):
                else_branch_nodes = [destination.node for destination in
                                     switch_node.out_port(0).get_connection().get_destinations()]
            if switch_node.is_out_port_connected(1):
                then_branch_nodes = [destination.node for destination in
                                     switch_node.out_port(1).get_connection().get_destinations()]
            data_flow_node = input_parameter_connection.node
            # parameter_name_id = '{0}:{1}'.format(data_flow_node.soft_get('id', data_flow_node.soft_get('name')),
            #                                      input_parameter_connection.idx)
            params_info[switch_node_id] = (input_parameter_connection, then_branch_nodes, else_branch_nodes,
                                           switch_node)

        results_info = {}
        for merge_node_id in merges_set:
            merge_node = Node(graph, merge_node_id)
            output_result_connection = merge_node.out_port(0).get_connection().get_destination()
            while output_result_connection.node.soft_get('op') == 'Identity':
                output_result_connection = output_result_connection.node.out_port(0). \
                    get_connection().get_destination()
            else_branch_node = None
            then_branch_node = None
            if merge_node.is_in_port_connected(0):
                else_branch_node = merge_node.in_port(0).get_connection().get_source().node
            if merge_node.is_in_port_connected(1):
                then_branch_node = merge_node.in_port(1).get_connection().get_source().node
            data_flow_node = output_result_connection.node
            # result_name_id = '{0}:{1}'.format(data_flow_node.soft_get('id', data_flow_node.soft_get('name')),
            #                                   output_result_connection.idx)
            results_info[merge_node_id] = (output_result_connection, then_branch_node, else_branch_node,
                                           merge_node)

        # create if node
        switch_id = list(switches_set)[0]
        if_name = '/'.join(str(switch_id).split('/')[0:-1])+'/if_node'
        if_node = If(graph, {'name': if_name}).create_node()
        condition_node_out_port.get_connection().add_destination(if_node.in_port(0))

        if_inputs = {}
        parameter_index = 1

        for param in params_info.values():
            param_value = param
            external_connection, then_branch_nodes, else_branch_nodes, switch_node = param_value
            switch_node.in_port(0).disconnect()
            switch_node.in_port(1).disconnect()
            if switch_node.is_out_port_connected(0):
                switch_node.out_port(0).disconnect()
            if switch_node.is_out_port_connected(1):
                switch_node.out_port(1).disconnect()
            switch_node['deprecated'] = True
            if_node.add_input_port(parameter_index, True)
            external_connection.get_connection().add_destination(if_node.in_port(parameter_index))
            if_inputs[parameter_index] = (then_branch_nodes, else_branch_nodes)
            parameter_index += 1

        if_outputs = {}
        result_index = 0

        for result in results_info.values():
            external_connection, then_branch_node, else_branch_node, merge_node = result
            merge_node.out_port(0).disconnect()
            if merge_node.is_in_port_connected(0):
                merge_node.in_port(0).disconnect()
            if merge_node.is_in_port_connected(1):
                merge_node.in_port(1).disconnect()
            if_node.add_output_port(result_index, True)
            external_connection.get_connection().set_source(if_node.out_port(result_index))

            if_outputs[result_index] = (then_branch_node, else_branch_node)
            result_index += 1

        # create then_branch

        then_graph = create_internal_graph(graph)
        else_graph = create_internal_graph(graph)
        then_graph.stage = 'front'
        else_graph.stage = 'front'

        if_node['then_graph'] = then_graph
        if_node['else_graph'] = else_graph
        then_graph_map = {}
        else_graph_map = {}
        then_graph_queue = Queue()
        else_graph_queue = Queue()
        for if_output_key in if_outputs.keys():
            then_branch_node, else_branch_node = if_outputs[if_output_key]
            if not then_branch_node is None:
                then_graph_queue.put(then_branch_node)
            if not else_branch_node is None:
                else_graph_queue.put(else_branch_node)

        then_graph_checked_nodes = set()
        while not then_graph_queue.empty():
            then_node_ext = then_graph_queue.get()
            then_node_ext_id = then_node_ext.soft_get('id', then_node_ext.soft_get('name'))
            if then_node_ext_id in then_graph_checked_nodes:
                continue
            then_graph_checked_nodes.add(then_node_ext_id)
            then_attrs = deepcopy(then_node_ext.attrs())
            then_attrs['name'] = then_node_ext_id + '/replaced'
            then_node = Op(then_graph, then_attrs).create_node()
            then_graph_map[then_node_ext_id] = then_node
            for in_port in then_node_ext.in_ports().values():
                if in_port.disconnected() or in_port.get_connection().get_source().disconnected() or \
                        in_port.get_connection().get_source() is None:
                    continue
                in_node = in_port.get_connection().get_source().node
                then_graph_queue.put(in_node)

        else_graph_checked_nodes = set()
        while not else_graph_queue.empty():
            else_node_ext = else_graph_queue.get()
            else_node_ext_id = else_node_ext.soft_get('id', else_node_ext.soft_get('name'))
            if else_node_ext_id in else_graph_checked_nodes:
                continue
            else_graph_checked_nodes.add(else_node_ext_id)
            else_attrs = deepcopy(else_node_ext.attrs())
            else_node = Op(else_graph, else_attrs).create_node()
            else_graph_map[else_node_ext_id] = else_node
            for in_port in else_node_ext.in_ports().values():
                if in_port.get_connection().get_source() is None:
                    continue
                in_node = in_port.get_connection().get_source().node
                else_graph_queue.put(in_node)

        for then_old_node_key in then_graph_map.keys():
            old_node = Node(graph, then_old_node_key)
            new_node = then_graph_map[then_old_node_key]
            for old_node_in_port in old_node.in_ports().values():
                in_id = old_node_in_port.idx
                old_node_connected_out_port = old_node_in_port.get_connection().get_source()
                if old_node_connected_out_port is None:
                    continue
                out_id = old_node_connected_out_port.idx
                old_connected_out_node = old_node_connected_out_port.node
                old_connected_out_node_id = old_connected_out_node. \
                    soft_get('id', old_connected_out_node.soft_get('name'))
                if not old_connected_out_node_id in then_graph_map.keys():
                    continue
                # neccessary copy edges attributes
                new_connected_node = then_graph_map[old_connected_out_node_id]
                new_node.add_input_port(in_id, True)
                new_connected_node.add_output_port(out_id, True)
                new_connected_node.out_port(out_id).get_connection().add_destination(new_node.in_port(in_id))

        for else_old_node_key in else_graph_map.keys():
            old_node = Node(graph, else_old_node_key)
            new_node = else_graph_map[else_old_node_key]
            for old_node_in_port in old_node.in_ports().values():
                in_id = old_node_in_port.idx
                old_node_connected_out_port = old_node_in_port.get_connection().get_source()
                if old_node_connected_out_port is None:
                    continue
                out_id = old_node_connected_out_port.idx
                old_connected_out_node = old_node_connected_out_port.node
                old_connected_out_node_id = old_connected_out_node. \
                    soft_get('id', old_connected_out_node.soft_get('name'))
                if not old_connected_out_node_id in else_graph_map.keys():
                    continue
                # neccessary copy edges attributes
                new_connected_node = else_graph_map[old_connected_out_node_id]
                new_node.add_input_port(in_id, True)
                new_connected_node.add_output_port(out_id, True)
                new_connected_node.out_port(out_id).get_connection().add_destination(new_node.in_port(in_id))

        for input_id in if_inputs.keys():
            then_nodes, else_nodes = if_inputs[input_id]
            then_map_keys = then_graph_map.keys()
            else_map_keys = else_graph_map.keys()

            if not then_nodes is None:
                for ext_then_node in then_nodes:
                    ext_then_id = ext_then_node.soft_get('id', ext_then_node.soft_get('name'))
                    if not ext_then_id in then_map_keys:
                        continue
                    then_node = then_graph_map[ext_then_id]
                    then_id = then_node.soft_get('id', then_node.soft_get('name'))
                    if then_node.has_and_set('parameter_template'):
                        parameter_id = '/'.join(then_id.split('/')[:-1])
                        param = Parameter(then_graph, {'name': parameter_id, 'input_id': input_id}).create_node()
                        param.add_output_port(0, True)
                        then_node.in_port(0).get_connection().set_source(param.out_port(0))

            if not else_nodes is None:
                for ext_else_node in else_nodes:
                    ext_else_id = ext_else_node.soft_get('id', ext_else_node.soft_get('name'))
                    if not ext_else_id in else_map_keys:
                        continue
                    else_node = else_graph_map[ext_else_id]
                    else_id = else_node.soft_get('id', else_node.soft_get('name'))
                    if else_node.has_and_set('parameter_template'):
                        parameter_id = '/'.join(else_id.split('/')[:-1])
                        param = Parameter(else_graph, {'name': parameter_id, 'input_id': input_id}).create_node()
                        param.add_output_port(0, True)
                        else_node.in_port(0).get_connection().set_source(param.out_port(0))

        for output_id in if_outputs.keys():
            ext_then_node, ext_else_node = if_outputs[output_id]
            ext_then_map_keys = then_graph_map.keys()
            ext_else_map_keys = else_graph_map.keys()

            assert not ext_then_node is None, "For output {0} cannot connect result node in then_body!".format(output_id)
            assert not ext_else_node is None, "For output {0} cannot connect result node in else_body!".format(output_id)

            ext_then_node_id = ext_then_node.soft_get('id', ext_then_node.soft_get('name'))
            ext_else_node_id = ext_else_node.soft_get('id', ext_else_node.soft_get('name'))
            if ext_then_node_id in ext_then_map_keys:
                then_node = then_graph_map[ext_then_node_id]
                then_id = then_node.soft_get('id', then_node.soft_get('name'))
                if then_node.has_and_set('result_template'):
                    result_id = '/'.join(then_id.split('/')[:-1])
                    result = Result(then_graph, {'name': result_id, 'output_id': output_id}).create_node()
                    result.add_input_port(0, True)
                    then_node.out_port(0).get_connection().set_destination(result.in_port(0))

            if ext_else_node_id in ext_else_map_keys:
                else_node = else_graph_map[ext_else_node_id]
                else_id = else_node.soft_get('id', else_node.soft_get('name'))
                if else_node.has_and_set('result_template'):
                    result_id = '/'.join(else_id.split('/')[:-1])
                    result = Result(else_graph, {'name': result_id, 'output_id': output_id}).create_node()
                    result.add_input_port(0, True)
                    else_node.out_port(0).get_connection().set_destination(result.in_port(0))

        print('gg')

        # for if_input_key in if_inputs.keys():
        #     then_branch_node, else_branch_node = if_inputs[if_input_key]
        #     if not then_branch_node is None:
        #         then_param = Parameter(then_graph, {'name':'{0}/{1}/Parameter_{2}'.\
        #                                format(if_name, 'then_branch', str(if_input_key))}).create_node()

    @staticmethod
    def convert_to_if_by_switch(switch_node, from_node_dict, to_node_dict):
        condition_node_out_port = switch_node.in_port(1).get_connection().get_source()
        condition_node = condition_node_out_port.node
        graph = condition_node.graph
        while condition_node.soft_get('op') == 'Identity':
            condition_node_out_port = condition_node.in_port(0).get_connection().get_source()
            condition_node = condition_node_out_port.node
        switches_set = SwitchMergeToIf.get_all_switches(condition_node, from_node_dict)
        merges_set = SwitchMergeToIf.get_all_merges(graph, switches_set, from_node_dict, to_node_dict)
        if len(merges_set) == 0:
            return

        SwitchMergeToIf.convert_to_if(graph, condition_node_out_port, switches_set, merges_set)
