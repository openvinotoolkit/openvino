# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging as log

from openvino.tools.mo.ops.loop import Loop
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.front.common.register_custom_ops import check_for_duplicates
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.extractor import extract_node_attrs
from openvino.tools.mo.front.onnx.extractor import onnx_op_extractor, onnx_op_extractors
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
from openvino.tools.mo.front.onnx.loader import node_id, add_initializers_and_inputs_to_graph
from openvino.tools.mo.graph.graph import Graph, Node, add_opoutput
from openvino.tools.mo.utils.error import Error


def create_edge_with_attrs(graph, src_name, src_internal_id, src_port, dst_id, dst_port):
    # src_name - name of input for edge
    # src_internal_id - input node dst_id. Can be the same as src_name or different if Parameter was created
    assert (graph.has_node(src_internal_id))
    edge_attrs = {
        'out': src_port,
        'in': dst_port,
        'name': src_name,
        'fw_tensor_debug_info': [(src_internal_id, src_name)],
        'in_attrs': ['in', 'name'],
        'out_attrs': ['out', 'name'],
        'data_attrs': ['fw_tensor_debug_info']
    }
    graph.add_edge(src_internal_id, dst_id, **edge_attrs)


def create_parameter_with_empty_attrs(graph, param_name):
    graph.add_node(param_name, kind='op', op='Parameter', name=param_name, pb=None, shape=None)
    parameter_node = Node(graph, param_name)
    # need to manually update necessary attrs for the node because extractor will not be called
    # for it because the node does not have .pb attribute
    Parameter.update_node_stat(parameter_node, {})
    parameter_node['internal_layer_id'] = len(graph.nodes)

    return parameter_node


def create_cross_body_edge(body_graph, external_edges, additional_params, src_internal_id, dst_id, dst_port):
    cur_graph = body_graph
    counter = 0
    is_finished = False
    transit_parameter = None
    # go through all levels of nested graphs starting from the deepest
    while not is_finished and 'parent_node' in cur_graph.graph:
        parent_graph = cur_graph.graph['parent_node'].graph
        external_edges.append([])
        additional_params.append({})
        assert 0 <= counter < len(additional_params)
        assert 0 <= counter < len(external_edges)
        # if parent graph contains input node, create edge from outer to inner graph
        if src_internal_id in parent_graph.graph['tensor_mapping']:
            log.debug('The edge between outer and inner graphs detected: {} -> {}'.format(src_internal_id, dst_id))
            # if parameter in inner graph already created, use it. Otherwise - create new one
            if parent_graph.graph['tensor_mapping'][src_internal_id] not in additional_params[counter - 1]:
                # possibly we create edge through several levels and have created transit parameter
                if transit_parameter is None:
                    # create new Parameter body node and connect the body node with the outer graph using it
                    param_id = str(src_internal_id)
                    parameter_node = create_parameter_with_empty_attrs(cur_graph, param_id)
                    src_id, src_port = param_id, 0
                else:
                    parameter_node = transit_parameter
                    src_id, src_port = transit_parameter.id, 0
                external_edges[counter].append((parent_graph.graph['tensor_mapping'][src_internal_id],
                                                parameter_node, src_internal_id))
                additional_params[counter][parent_graph.graph['tensor_mapping'][src_internal_id][0]] = parameter_node
            else:
                src_id, src_port = additional_params[counter - 1][parent_graph.graph['tensor_mapping'][src_internal_id][0]].id, 0
            is_finished = True
        else:
            # check that we are not in process of creating edge through several borders
            # if we have transit node, it becomes destination of edge
            # otherwise create new Parameter
            if transit_parameter is None:
                # create new Parameter in inner graph in hope that we will find node later
                param_id = str(src_internal_id).split(':')[0]
                parameter_node = create_parameter_with_empty_attrs(cur_graph, param_id)
            else:
                parameter_node = transit_parameter
                param_id = transit_parameter.id

            # create transit parameter in outer graph in hope that real input will be found later
            parent_param_id = str(src_internal_id).split(':')[0] + "_transit"
            parent_parameter_node = create_parameter_with_empty_attrs(parent_graph, parent_param_id)

            external_edges[counter].append(((parent_param_id, 0), parameter_node, parent_param_id))
            src_id, src_port = param_id, 0
            additional_params[counter][parent_param_id + ":0"] = parameter_node
            transit_parameter = parent_parameter_node

        if cur_graph.has_node(dst_id):
            create_edge_with_attrs(cur_graph, src_internal_id, src_id, src_port, dst_id, dst_port)

        cur_graph = parent_graph
        counter += 1

    return is_finished


class LoopExtractor(FrontExtractorOp):
    op = 'Loop'
    enabled = True

    @classmethod
    def extract(cls, loop_node):
        Loop.update_node_stat(loop_node, {})

        body_graph_proto = onnx_attr(loop_node, 'body', 'g', None)
        main_graph = loop_node.graph

        # create a Graph object for the body and take graph attributes from the main graph
        body_graph = Graph()
        main_graph_attrs_copy = {}
        for attr_key, attr_value in main_graph.graph.items():
            if attr_key not in ['tensor_mapping', 'parent_node']:
                main_graph_attrs_copy[attr_key] = copy.deepcopy(attr_value)
        body_graph.graph.update(main_graph_attrs_copy)
        loop_node['body'] = body_graph
        # save parent node for nested loops to know which node contains body (and which graph is on upper level)
        body_graph.graph['parent_node'] = loop_node

        # maps a tensor name to a node produced it and the node port: str -> (node_id, node_port)
        data_nodes_map = {}
        body_graph.graph['tensor_mapping'] = data_nodes_map  # save mapping for possible Loop inside the Loop

        body_parameters = add_initializers_and_inputs_to_graph(body_graph, body_graph_proto, data_nodes_map)

        external_edges = []  # (src_node, src_out_port), dest_body_parameter_node
        # save additional edges information for graph on each level, the first one is the deepest
        additional_params = []  # (src_node, src_out_port) -> parameter_node (for manually added Parameters)
        # Go through all nodes in the original model order because data nodes are defined on-the-fly and order matters
        for pb_node in body_graph_proto.node:
            # create an NX node
            id = body_graph.unique_id(node_id(pb_node))
            body_graph.add_node(id, pb=pb_node, kind='op')
            if hasattr(body_graph, 'op_names_statistic') and hasattr(pb_node, 'op_type'):
                body_graph.op_names_statistic[pb_node.op_type] += 1

            # add incoming edges based on data_nodes_map
            for dst_port, inp in enumerate(pb_node.input):
                # should add edge src_internal_id --> dst_id
                if inp not in data_nodes_map:
                    if inp == '':
                        # input is omitted; most likely it corresponds to an optional input for an operator
                        continue
                    else:
                        is_finished = create_cross_body_edge(body_graph, external_edges, additional_params,
                                                             inp, id, dst_port)
                        if not is_finished:
                            raise Error(
                                'Reference to "{}" is not satisfied. A node refer not existing data tensor. ONNX '
                                'model is not consistent. Protobuf fragment: {}', inp, pb_node)
                else:
                    src_id, src_port = data_nodes_map[inp]
                    create_edge_with_attrs(body_graph, inp, src_id, src_port, id, dst_port)

            # add outgoing edges to data_nodes_map
            for src_port, out in enumerate(pb_node.output):
                if out in data_nodes_map:
                    log.debug("Detected reuse of blob {}.".format(out))
                data_nodes_map[out] = (id, src_port)

        body_results = []
        for output in body_graph_proto.output:
            tensor_name = str(output.name)
            node_name, output_port = data_nodes_map[tensor_name]
            assert body_graph.has_node(node_name), 'The body graph does not contain output with name "{}"'.format(
                node_name)
            body_results.append(Node(body_graph, add_opoutput(body_graph, node_name, output_port, False)))

        # add 'internal_layer_id' attribute which is a must have attribute for the loop body node
        for idx, body_node in enumerate(body_graph.get_op_nodes()):
            body_node['internal_layer_id'] = idx

        loop_carried_dependencies_count = len(body_graph_proto.input) - 2
        scan_outputs_count = len(body_graph_proto.output) - 1 - loop_carried_dependencies_count

        # Loop inputs:
        #   0 - trip count
        #   1 - execution condition
        #   2 .. - loop carried dependencies

        # Loop outputs:
        #   0 .. loop_carried_dependencies_count - 1 - loop carried dependencies
        #   loop_carried_dependencies_count .. - scan outputs

        # Body inputs:
        #   0 - iteration number
        #   1 - execution condition
        #   2 .. - loop carried dependencies

        # Body outputs:
        #   0 - execution condition
        #   1 .. loop_carried_dependencies_count - loop carried dependencies
        #   loop_carried_dependencies_count + 1 .. - scan outputs

        # some of the inputs/outputs may not be connected but the normalization transformation will take care of it
        # connection Loop body nodes with external input edges
        next_loop_input_port_idx = sorted(loop_node.in_edges().keys())[-1] + 1
        cur_graph = body_graph
        for external_edges_subg in external_edges:
            if 'parent_node' not in cur_graph.graph:
                continue
            cur_loop_node = cur_graph.graph['parent_node']
            parent_graph = cur_loop_node.graph
            for (src_node, src_port), body_node, tensor_name in external_edges_subg:
                create_edge_with_attrs(parent_graph, tensor_name, src_node, src_port,
                                       cur_loop_node.id, next_loop_input_port_idx)

                Loop.connect_body_input(cur_loop_node, next_loop_input_port_idx, body_node)
                next_loop_input_port_idx += 1
            cur_graph = parent_graph

        # mark current iteration input Parameter node
        Loop.mark_current_iteration_parameter_node(loop_node, body_parameters[0])

        # connect initial value for "execution condition" input of the loop
        Loop.connect_body_input(loop_node, 1, body_parameters[1])
        # add back edge with "execution condition"
        Loop.add_back_edge(loop_node, body_parameters[1], body_results[0])
        # mark "execution condition" Result node
        Loop.mark_execution_condition_result_node(loop_node, body_results[0])

        # connect initial value for "loop carried" dependencies variables
        for idx in range(loop_carried_dependencies_count):
            Loop.connect_body_input(loop_node, idx + 2, body_parameters[idx + 2])
        # add back edge for "loop carried" dependencies variables
        for idx in range(loop_carried_dependencies_count):
            Loop.add_back_edge(loop_node, body_parameters[idx + 2], body_results[idx + 1])
        # connect final value for "loop carried" dependencies variables
        for idx in range(loop_carried_dependencies_count):
            Loop.connect_body_output(loop_node, idx, body_results[idx + 1])

        # connect "scan outputs" and mark axis for concatenation
        for idx in range(loop_carried_dependencies_count, loop_carried_dependencies_count + scan_outputs_count):
            Loop.connect_body_output(loop_node, idx, body_results[idx + 1], axis=0)

        # run function to parse body nodes attributes similar to the main graph
        extract_node_attrs(body_graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))
        return cls.enabled
