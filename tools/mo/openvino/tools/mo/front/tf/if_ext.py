# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.If import If
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.front.common.register_custom_ops import check_for_duplicates
from openvino.tools.mo.front.extractor import FrontExtractorOp, extract_node_attrs
from openvino.tools.mo.front.tf.extractor import tf_op_extractor, tf_op_extractors
from openvino.tools.mo.front.tf.extractors.subgraph_utils import update_body_graph, convert_graph_inputs_to_parameters, \
    get_graph_proto, create_internal_graph
from openvino.tools.mo.graph.graph import Node, Graph


def extract_if(cls, if_node: Node):
    If.update_node_stat(if_node, {})

    # check that required body and condition functions exist in the graph library
    main_graph = if_node.graph
    then_graph_proto = get_graph_proto(main_graph, 'then_branch', if_node)
    else_graph_proto = get_graph_proto(main_graph, 'else_branch', if_node)

    then_graph = create_internal_graph(main_graph)
    if_node['then_graph'] = then_graph

    else_graph = create_internal_graph(main_graph)
    if_node['else_graph'] = else_graph

    # create Parameter nodes for the then/else graphs
    for input_index, (body_graph, body_graph_proto) in enumerate(zip((then_graph, else_graph), (then_graph_proto,
                                                                                                else_graph_proto))):

        body_parameters, body_parameter_names = convert_graph_inputs_to_parameters(body_graph, body_graph_proto)

        # update the If body graph with the body function graph
        body_results = []
        update_body_graph(body_graph, body_graph_proto, body_parameter_names, body_results)

        body_graph.stage = 'front'

        # connect external input ports with body parameter nodes except input with condition
        for idx in range(0, len(body_parameters)):
            If.connect_body_input(if_node, not input_index, idx + 1, body_parameters[idx])

        # connect body outputs with If operation output ports
        for idx in range(len(body_results)):
            If.connect_body_output(if_node, not input_index, idx, body_results[idx])

        # run function to parse body nodes attributes similar to the main graph
        extract_node_attrs(body_graph, lambda node: tf_op_extractor(node, check_for_duplicates(tf_op_extractors)))

    return cls.enabled


class IfExtractor(FrontExtractorOp):
    op = 'If'
    enabled = True

    @classmethod
    def extract(cls, if_node: Node):
        return extract_if(cls, if_node)


class StatelessIfExtractor(FrontExtractorOp):
    op = 'StatelessIf'
    enabled = True

    @classmethod
    def extract(cls, if_node: Node):
        return extract_if(cls, if_node)
