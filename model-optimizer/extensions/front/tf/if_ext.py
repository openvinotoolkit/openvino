# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

from extensions.front.tf.while_ext import update_body_graph
from extensions.ops.If import If
from extensions.ops.parameter import Parameter
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.extractor import FrontExtractorOp, extract_node_attrs
from mo.front.tf.extractor import tf_op_extractor, tf_op_extractors
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.graph.graph import Node, Graph
from mo.ops.op import PermuteAttrs


def extract_method(cls, if_node: Node):
    If.update_node_stat(if_node, {})

    if_name = if_node.soft_get('name', if_node.id)

    # check that required body and condition functions exist in the graph library
    main_graph = if_node.graph
    then_graph_name = if_node.pb.attr['then_branch'].func.name
    else_graph_name = if_node.pb.attr['else_branch'].func.name
    assert 'library' in main_graph.graph, 'The graph does not contain a library that is required ' \
                                          'by node with name "{}".'.format(if_name)
    library_graph = main_graph.graph['library']

    assert then_graph_name in library_graph, 'The library does not contain a function with name "{}" ' \
                                             'that is required by node ' \
                                             'with name "{}".'.format(then_graph_name, if_name)
    then_graph_proto = library_graph[then_graph_name]

    assert else_graph_name in library_graph, 'The library does not contain a function with name "{}" ' \
                                             'that is required by node ' \
                                             'with name "{}".'.format(else_graph_name, if_name)
    else_graph_proto = library_graph[else_graph_name]

    # create "then" graph
    then_graph = Graph()
    # fill the body graph
    for attr_key in main_graph.graph.keys():
        if attr_key != 'library':
            then_graph.graph[attr_key] = copy.deepcopy(main_graph.graph[attr_key])
        else:
            # it is sufficient to have a link to the library
            then_graph.graph['library'] = main_graph.graph['library']
    if_node['then_graph'] = then_graph

    # create "else" graph
    else_graph = Graph()
    # fill the body graph
    for attr_key in main_graph.graph.keys():
        if attr_key != 'library':
            else_graph.graph[attr_key] = copy.deepcopy(main_graph.graph[attr_key])
        else:
            # it is sufficient to have a link to the library
            else_graph.graph['library'] = main_graph.graph['library']
    if_node['else_graph'] = else_graph

    # create Parameter nodes for the then/else graphs
    for input_index, (body_graph, body_graph_proto) in enumerate(zip((then_graph, else_graph), (then_graph_proto,
                                                                                                else_graph_proto))):
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
        return extract_method(cls, if_node)


class StatelessIfExtractor(FrontExtractorOp):
    op = 'StatelessIf'
    enabled = True

    @classmethod
    def extract(cls, if_node: Node):
        return extract_method(cls, if_node)
