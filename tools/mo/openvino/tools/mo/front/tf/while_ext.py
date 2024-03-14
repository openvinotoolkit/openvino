# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.loop import Loop
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.front.common.register_custom_ops import check_for_duplicates
from openvino.tools.mo.front.extractor import extract_node_attrs, FrontExtractorOp
from openvino.tools.mo.front.tf.extractor import tf_op_extractor, tf_op_extractors, create_tf_edge
from openvino.tools.mo.front.tf.extractors.subgraph_utils import update_body_graph, convert_graph_inputs_to_parameters, \
    get_graph_proto, create_internal_graph
from openvino.tools.mo.graph.graph import add_opoutput, Graph, Node


class WhileExtractor(FrontExtractorOp):
    """
    The While operation is a variation of the while_loop primitive from TensorFlow 2 Python API.
    While can have stateful operations in the body and condition graphs that does not influence on inference so
    the logic for handling While and StatelessWhile (see below) is the same.
    """
    op = 'While'
    enabled = True

    @classmethod
    def extract(cls, loop_node):
        Loop.update_node_stat(loop_node, {})

        # check that required body and condition functions exist in the graph library
        main_graph = loop_node.graph
        body_graph_proto = get_graph_proto(main_graph, 'body', loop_node)
        cond_graph_proto = get_graph_proto(main_graph, 'cond', loop_node)

        body_graph = create_internal_graph(main_graph)
        loop_node['body'] = body_graph
        # create Parameter nodes for the body graph
        body_parameters, body_parameter_names = convert_graph_inputs_to_parameters(body_graph, body_graph_proto)

        # update the loop body graph with the body function graph
        body_results = []
        update_body_graph(body_graph, body_graph_proto, body_parameter_names, body_results)

        # update the loop body graph with the condition function graph
        update_body_graph(body_graph, cond_graph_proto, body_parameter_names, body_results)

        # add 'internal_layer_id' attribute which is a must have attribute for the loop body node
        for idx, body_node in enumerate(body_graph.get_op_nodes()):
            body_node['internal_layer_id'] = idx

        body_graph.stage = 'front'

        # Currently,
        # Loop Inputs Order:
        #   0    - current iteration
        #   1    - trip count
        #   2..  - "loop carried" dependencies variables
        #
        # Body Inputs Order:
        #   0    - current iteration
        #   1    - trip count
        #   2..  - "loop carried" dependencies variables
        #
        # Body Outputs Order:
        #   0      - current iteration
        #   1      - trip count
        #   2..    - "loop carried" dependencies variables
        #
        # Loop Outputs Order:
        #   0    - current iteration
        #   1    - trip count
        #   2..  - "loop carried" dependencies variables
        #
        # so inputs must be reordered and execution condition must be created in the front transformation
        # to be aligned with the specification

        # connect external input ports with body parameter nodes except current iteration
        # since it must be disconnected from external port
        for idx in range(1, len(body_parameters)):
            Loop.connect_body_input(loop_node, idx, body_parameters[idx])

        # mark current iteration input Parameter node and execution condition Result node
        Loop.mark_current_iteration_parameter_node(loop_node, body_parameters[0])
        Loop.mark_execution_condition_result_node(loop_node, body_results[-1])

        # connect back edges in the body except current iteration
        for idx in range(1, len(body_parameters)):
            Loop.add_back_edge(loop_node, body_parameters[idx], body_results[idx])

        # connect body outputs with Loop operation output ports except the execution condition result
        for idx in range(len(body_results) - 1):
            Loop.connect_body_output(loop_node, idx, body_results[idx])

        # run function to parse body nodes attributes similar to the main graph
        extract_node_attrs(body_graph, lambda node: tf_op_extractor(node, check_for_duplicates(tf_op_extractors)))
        return cls.enabled


class StatelessWhileExtractor(FrontExtractorOp):
    """
    The StatelessWhile operation is a variation of the while_loop primitive from TensorFlow 2 Python API.
    StatelessWhile does not have stateful operations in the body and condition graphs.
    """
    op = 'StatelessWhile'
    enabled = True

    @classmethod
    def extract(cls, loop_node):
        WhileExtractor.extract(loop_node)
        return cls.enabled
