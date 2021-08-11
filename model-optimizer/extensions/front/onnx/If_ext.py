# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging as log

from extensions.ops.If import If
from extensions.ops.parameter import Parameter
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.extractor import FrontExtractorOp, extract_node_attrs
from mo.front.onnx.extractor import onnx_op_extractor, onnx_op_extractors
from mo.front.onnx.extractors.utils import onnx_attr
from mo.front.onnx.loader import node_id, add_initializers_and_inputs_to_graph
from mo.graph.graph import Node, Graph, add_opoutput
from mo.ops.op import PermuteAttrs
from mo.utils.graph import node_incoming_neighbourhood


class IfExtractor(FrontExtractorOp):
    op = 'If'
    enabled = True

    @staticmethod
    def extract_body(if_node, body_name):

        body_graph_proto = onnx_attr(if_node, body_name, 'g', None)
        main_graph = if_node.graph

        # create a Graph object for the body and take graph attributes from the main graph
        body_graph = Graph()
        main_graph_attrs_copy = copy.deepcopy(main_graph.graph)
        del main_graph_attrs_copy['tensor_mapping']
        body_graph.graph.update(main_graph_attrs_copy)
        graph_name = 'then_graph' if body_name == 'then_branch' else 'else_graph'
        if_node[graph_name] = body_graph

        # maps a tensor name to a node produced it and the node port: str -> (node_id, node_port)
        data_nodes_map = {}
        body_graph.graph['tensor_mapping'] = data_nodes_map  # save mapping for possible Loop inside the Loop

        body_parameters = add_initializers_and_inputs_to_graph(body_graph, body_graph_proto, data_nodes_map)

        external_edges = []  # (src_node, src_out_port), dest_body_parameter_node
        additional_params = {}  # (src_node, src_out_port) -> parameter_node (for manually added Parameters)
        # Go through all nodes in the original model order because data nodes are defined on-the-fly and order matters
        for pb_node in body_graph_proto.node:
            # create an NX node
            id = body_graph.unique_id(node_id(pb_node))
            body_graph.add_node(id, pb=pb_node, kind='op')

            # add incoming edges based on data_nodes_map
            for dst_port, inp in enumerate(pb_node.input):
                # should add edge inp --> id
                if inp not in data_nodes_map:
                    if inp == '':
                        # input is omitted; most likely it corresponds to an optional input for an operator
                        continue
                    elif inp in main_graph.graph['tensor_mapping']:
                        log.debug('The edge between outer and inner graphs detected: {} -> {}'.format(inp, id))
                        if main_graph.graph['tensor_mapping'][inp] not in additional_params:
                            # create new Parameter body node and connect the body node with the outer graph using it
                            param_id = str(inp)
                            body_graph.add_node(param_id, kind='op', op='Parameter', name=param_id, pb=None, shape=None)
                            parameter_node = Node(body_graph, param_id)
                            # need to manually update necessary attrs for the node because extractor will not be called
                            # for it because the node does not have .pb attribute
                            Parameter.update_node_stat(parameter_node, {})
                            external_edges.append((main_graph.graph['tensor_mapping'][inp], parameter_node, inp))
                            src_id, src_port = param_id, 0
                            additional_params[main_graph.graph['tensor_mapping'][inp]] = parameter_node
                        else:
                            src_id, src_port = additional_params[main_graph.graph['tensor_mapping'][inp]].id, 0
                    else:
                        raise Error('Reference to "{}" is not satisfied. A node refer not existing data tensor. ONNX '
                                    'model is not consistent. Protobuf fragment: {}', inp, pb_node)
                else:
                    src_id, src_port = data_nodes_map[inp]

                assert (body_graph.has_node(src_id))
                edge_attrs = {
                    'out': src_port,
                    'in': dst_port,
                    'name': inp,
                    'fw_tensor_debug_info': [(src_id, inp)],
                    'in_attrs': ['in', 'name'],
                    'out_attrs': ['out', 'name'],
                    'data_attrs': ['fw_tensor_debug_info']
                }
                body_graph.add_edge(src_id, id, **edge_attrs)

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

        body_graph.stage = 'front'
        next_if_input_port_idx = sorted(if_node.in_edges().keys())[-1] + 1
        condition = body_name == 'then_branch'
        for (src_node, src_port), body_node, tensor_name in external_edges:
            main_graph.add_edge(src_node, if_node.id, **{'out': src_port,
                                                         'in': next_if_input_port_idx,
                                                         'name': src_node,
                                                         'fw_tensor_debug_info': [(src_node, tensor_name)],
                                                         'in_attrs': ['in', 'name'],
                                                         'out_attrs': ['out', 'name'],
                                                         'data_attrs': ['fw_tensor_debug_info']}
                                )
            If.connect_body_input(if_node, condition, next_if_input_port_idx, body_node)
            next_if_input_port_idx += 1

        for idx, out_node in enumerate(body_results):
            If.connect_body_output(if_node, condition, idx, out_node)

        extract_node_attrs(body_graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))

    @classmethod
    def extract(cls, if_node):
        If.update_node_stat(if_node, {})
        IfExtractor.extract_body(if_node, 'then_branch')
        IfExtractor.extract_body(if_node, 'else_branch')
        return cls.enabled
