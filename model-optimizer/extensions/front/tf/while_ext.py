# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

from extensions.ops.loop import Loop
from extensions.ops.parameter import Parameter
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.extractor import extract_node_attrs, FrontExtractorOp
from mo.front.tf.extractor import tf_op_extractor, tf_op_extractors, create_tf_edge
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.graph.graph import add_opoutput, Graph, Node
from mo.ops.op import PermuteAttrs


def update_body_graph(body_graph: Graph, subgraph_proto: dict,
                      body_parameter_names: list, body_results: list):
    """
    Updates the loop body graph with a sub-graph (for body or condition functions)
    :param body_graph: a loop body graph to be updated
    :param subgraph_proto: a sub-graph in a protobuf format to be added into the loop body graph
    :param body_parameter_names: a (unchanged) list of parameters in the loop body graph
    :param body_results: a list of Result nodes that is extended with a list from a sub-graph
    """
    # create a map from a node name in original model to a name in a loop body graph assuming
    # that names in the original model are unique
    # initially, the map contains names for parameters that are common for the body and condition graphs
    map_original_name = {}
    for idx, pb_node in enumerate(subgraph_proto['input_arg']):
        map_original_name[pb_node.name] = body_parameter_names[idx]

    # walk through all nodes (non-parameter and non-result nodes) and add into the loop body graph
    for pb_node in subgraph_proto['node_def']:
        # create an NX node
        id = body_graph.unique_id(pb_node.name)
        map_original_name[pb_node.name] = id
        body_graph.add_node(id, pb=pb_node, kind='op')
        if hasattr(body_graph, 'op_names_statistic') and hasattr(pb_node, 'op'):
            body_graph.op_names_statistic[pb_node.op] += 1

        # add incoming edges based on data_nodes_map
        for dst_port, inp in enumerate(pb_node.input):
            orig_src_id = inp.split(":")[0]

            # TODO: avoid this temporal workaround for TF 2.4 or higher RNN layers:
            #  skip control flow dependency
            if orig_src_id[0] == '^':
                continue

            src_id = map_original_name[orig_src_id]
            src_port = 0 if len(inp.split(":")) == 1 else int(inp.split(":")[-1])
            assert (body_graph.has_node(src_id))

            body_graph.add_edges_from([create_tf_edge(src_id + ":" + str(src_port), id, dst_port)])

    # create Result nodes in the loop body graph
    for output in subgraph_proto['output_arg']:
        output_name = subgraph_proto['ret'][output.name]
        orig_src_id = output_name.split(":")[0]
        src_id = map_original_name[orig_src_id]
        src_port = 0 if len(output_name.split(":")) == 1\
            else int(output_name.split(":")[-1])
        assert body_graph.has_node(src_id), 'The body graph does not contain output with name "{}"'.format(
            src_id)
        body_results.append(Node(body_graph, add_opoutput(body_graph, src_id, src_port, False)))


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
        loop_name = loop_node.soft_get('name', loop_node.id)

        # check that required body and condition functions exist in the graph library
        main_graph = loop_node.graph
        body_graph_name = loop_node.pb.attr['body'].func.name
        cond_graph_name = loop_node.pb.attr['cond'].func.name
        assert 'library' in main_graph.graph, 'The graph does not contain a library that is required ' \
                                              'by node with name "{}".'.format(loop_name)
        library_graph = main_graph.graph['library']

        assert body_graph_name in library_graph, 'The library does not contain a function with name "{}" ' \
                                                 'that is required by node ' \
                                                 'with name "{}".'.format(body_graph_name, loop_name)
        body_graph_proto = library_graph[body_graph_name]

        assert cond_graph_name in library_graph, 'The library does not contain a function with name "{}" ' \
                                                 'that is required by node ' \
                                                 'with name "{}".'.format(cond_graph_name, loop_name)
        cond_graph_proto = library_graph[cond_graph_name]

        body_graph = Graph()
        # fill the body graph
        for attr_key in main_graph.graph.keys():
            if attr_key != 'library':
                body_graph.graph[attr_key] = copy.deepcopy(main_graph.graph[attr_key])
            else:
                # it is sufficient to have a link to the library
                body_graph.graph['library'] = main_graph.graph['library']
        loop_node['body'] = body_graph

        # create Parameter nodes for the body graph
        body_parameters = []
        body_parameter_names = []
        for idx, pb_node in enumerate(body_graph_proto['input_arg']):
            param_id = body_graph.unique_id(pb_node.name)
            body_graph.add_node(param_id, name=param_id, kind='op', op='Parameter', pb=None, shape=None)
            parameter_node = Node(body_graph, pb_node.name)
            Parameter.update_node_stat(parameter_node,
                                       {'data_type': tf_dtype_extractor(pb_node.type),
                                        'permute_attrs': PermuteAttrs().update_attrs(attrs=[('shape', 'output:0')])}
                                       )
            body_parameters.append(parameter_node)
            body_parameter_names.append(param_id)

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
        for idx in range(len(body_results)-1):
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
