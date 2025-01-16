# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.front.tf.extractor import tf_op_extractor, tf_op_extractors, create_tf_edge
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor
from openvino.tools.mo.graph.graph import Graph, Node, add_opoutput
from openvino.tools.mo.ops.op import PermuteAttrs


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
        src_port = 0 if len(output_name.split(":")) == 1 \
            else int(output_name.split(":")[-1])
        assert body_graph.has_node(src_id), 'The body graph does not contain output with name "{}"'.format(
            src_id)
        body_results.append(Node(body_graph, add_opoutput(body_graph, src_id, src_port, False)))

    return True


def get_graph_proto(external_graph: Graph, graph_id: str, node_with_graph: Node):
    graph_name = node_with_graph.pb.attr[graph_id].func.name
    node_name = node_with_graph.soft_get('name', node_with_graph.id)

    assert 'library' in external_graph.graph, 'The graph does not contain a library that is required ' \
                                              'by node with name "{}".'.format(node_name)

    library_graph = external_graph.graph['library']

    assert graph_name in library_graph, 'The library does not contain a function with name "{}" ' \
                                        'that is required by node ' \
                                        'with name "{}".'.format(graph_name, node_name)
    return library_graph[graph_name]


def create_internal_graph(external_graph: Graph):
    internal_graph = Graph()
    # fill the body graph
    for attr_key in external_graph.graph.keys():
        if attr_key != 'library':
            internal_graph.graph[attr_key] = copy.deepcopy(external_graph.graph[attr_key])
        else:
            # it is sufficient to have a link to the library
            internal_graph.graph['library'] = external_graph.graph['library']
    return internal_graph


def convert_graph_inputs_to_parameters(internal_graph, internal_graph_proto):
    # create Parameter nodes for the body graph
    body_parameters = []
    body_parameter_names = []
    for idx, pb_node in enumerate(internal_graph_proto['input_arg']):
        param_id = internal_graph.unique_id(pb_node.name)
        internal_graph.add_node(param_id, name=param_id, kind='op', op='Parameter', pb=None, shape=None)
        parameter_node = Node(internal_graph, pb_node.name)
        Parameter.update_node_stat(parameter_node,
                                   {'data_type': tf_dtype_extractor(pb_node.type),
                                    'permute_attrs': PermuteAttrs().update_attrs(attrs=[('shape', 'output:0')])}
                                   )
        body_parameters.append(parameter_node)
        body_parameter_names.append(param_id)
    return body_parameters, body_parameter_names
