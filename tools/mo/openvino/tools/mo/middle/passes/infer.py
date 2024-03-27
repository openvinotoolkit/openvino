# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from typing import List

import networkx as nx

from openvino.tools.mo.front.common.layout import get_dim_from_layout
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension
from openvino.tools.mo.graph.graph import Node, Graph, dict_includes
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg, shrink_str_value


def log_debug_dict(nodes_per_port: dict, direction_name: str):
    for port, node in nodes_per_port.items():
        value = shrink_str_value(node.soft_get('value'))
        log.debug('{}[{}]: shape = {}, value = {}'.format(direction_name, port, node.soft_get('shape'), value))


def control_flow_infer(graph: Graph, node_name: str):
    """
       Executes constant control flow. Propagates nodes executability
    """
    if graph.node[node_name]['kind'] == 'data':
        return

    def mark_executability(node_id: str, is_executable: bool):
        if is_executable and not graph.node[node_id]['executable']:
            return
        graph.node[node_id]['executable'] = is_executable

    in_edges_with_data = graph.in_edges(node_name, data=True)
    in_df_edges_with_data = [(u, v, attrs) for u, v, attrs in in_edges_with_data
                             if 'control_flow_edge' not in attrs or not attrs['control_flow_edge']]
    in_cf_edges_with_data = [(u, v, attrs) for u, v, attrs in in_edges_with_data
                             if 'control_flow_edge' in attrs and attrs['control_flow_edge']]
    is_executable_df = all([graph.node[u]['executable'] for u, _, attrs in in_df_edges_with_data]
                           if len(in_df_edges_with_data) else [True])
    is_executable_cf = all([graph.node[u]['executable'] for u, _, attrs in in_cf_edges_with_data]
                           if len(in_cf_edges_with_data) else [True])
    is_executable = is_executable_df and is_executable_cf

    node = Node(graph, node_name)
    if 'cf_infer' in graph.node[node_name] and callable(node.cf_infer):
        node.cf_infer(node, is_executable, mark_executability)
    else:
        for _, out_data in graph.out_edges(node_name):
            mark_executability(out_data, is_executable)


def exit_bound_edges(graph: Graph, sources: list, end_node_attrs: dict):
    """
    Finds all descendant nodes for each node from 'sources' that have given attributes from end_node_attrs.
    For each found node, create a tuple with a given element from 'source' and the node.
    """
    result = []
    for node in sources:
        for end_node in nx.descendants(graph, node):
            if dict_includes(big=graph.node[end_node], sub_dict=end_node_attrs):
                result.append((node, end_node, 0, {}))
    return result


def partial_infer(graph: Graph, start_node: str = None):
    """
    Tries to execute constant parts of the graph and deduce as much as possible
    information following the data flow, e.g. calculate and propagate shapes and
    constant values. Partially or completely defined values are stored in data
    nodes (kind='data').
    """
    # We have to turn off strict mode due to above we add and remove edeges without attributes that is prohibited
    graph.strict_mode = False
    cycle_nodes = graph.get_nodes_with_attributes(is_cyclic=True)
    cycle_nodes = [Node(graph, node).out_node().id for node in cycle_nodes]
    ebunch_cyclic = list(graph.out_edges(nbunch=cycle_nodes, data=True, keys=True))
    ebunch_reconnected = exit_bound_edges(graph, sources=cycle_nodes, end_node_attrs={'op': 'Exit'})
    graph.remove_edges_from(ebunch_cyclic)
    graph.add_edges_from(ebunch_reconnected)

    try:
        nodes = list(nx.topological_sort(graph))
    except:
        raise Error('Graph contains a cycle. Can not proceed. ' + refer_to_faq_msg(97))

    graph.remove_edges_from(ebunch_reconnected)
    graph.add_edges_from(ebunch_cyclic)
    graph.strict_mode = True

    # Mark all nodes as not inferred yet
    if start_node is not None:
        start_index = nodes.index(start_node)
        nx.set_node_attributes(G=graph.subgraph(nodes[start_index:]), name='is_partial_inferred', values=False)
    else:
        nx.set_node_attributes(G=graph, name='is_partial_inferred', values=False)

    nx.set_node_attributes(G=graph, name='executable',
                           values={n: True for n in graph.get_nodes_with_attributes(kind='data')})

    # first we infer constant sub-graphs so the reverse infer could use constant values sub-graphs. For example,
    # convolution weights may be reshuffled by some operation in the graph and are not directly consumed by the conv
    # node
    infer_nodes(graph, nodes, True)

    # we may need to deduce shape for Parameter node(s) if it is not defined
    need_reverse_infer = False
    for parameter in graph.get_op_nodes(op='Parameter'):
        if parameter.soft_get('shape', None) is None:
            need_reverse_infer = True

    if need_reverse_infer:
        reverse_infer(graph, nodes)

    infer_nodes(graph, nodes, False)

    not_fully_inferred = graph.get_nodes_with_attributes(is_not_fully_inferred=True)
    for n in not_fully_inferred:
        node = Node(graph, n)
        if node.has_and_set('infer'):
            node.infer(node)

    return graph


def infer_nodes(graph: Graph, nodes: List[Node], constant_subgraph_only: bool = False):
    """
    Run "infer" function of the specified nodes.

    :param graph: graph with nodes
    :param nodes: list of node ids in the topological order
    :param constant_subgraph_only: flag which specifies whether only inference of constant sub-graphs should be done
    """
    debug_logger = log.getLogger().isEnabledFor(log.DEBUG)
    for n in nodes:
        # Data Flow Infer
        node = Node(graph, n)
        node_name = node.soft_get('name', node.id)
        try:
            if node.has('is_partial_inferred') and not node.is_partial_inferred:
                if node.has('infer') and not node.infer is None:
                    # we consider that operation will produce value if all inputs are constants or it is
                    # 'ShapeOf' operation
                    if constant_subgraph_only:
                        in_values = [port.data.get_value() for port in node.in_ports().values()]
                        if node.soft_get('op') == 'Parameter' or any(value is None for value in in_values) or \
                                (node.soft_get('op') == 'ShapeOf' and node.in_port(0).data.get_shape() is None):
                            # if here will be any new ShapeOf type operation, we should update condition above
                            continue

                    if debug_logger:
                        log.debug('-' * 20)
                        log.debug('Partial infer for {}'.format(node.soft_get('name')))
                        log.debug('Op: {}'.format(node.soft_get('op')))
                        log.debug('Inputs:')
                        log_debug_dict(node.in_nodes(), 'input')

                    node.infer(node)
                    out_nodes = node.out_nodes()

                    # propagate nchw_layout attributes to data nodes
                    if node.has('nchw_layout'):
                        for out_node in out_nodes.values():
                            out_node['nchw_layout'] = node.nchw_layout

                    # In debug print current node attributes, input shapes/values and output shape/values
                    if debug_logger:
                        log.debug('Outputs:')
                        log_debug_dict(node.out_nodes(), 'output')

                    if not constant_subgraph_only:
                        not_all_output_shapes = False

                        for out_port, out_node in out_nodes.items():
                            not_all_output_shapes = False
                            if not out_node.has_valid('shape'):
                                log.error('Shape is not defined for output {} of "{}".'.format(out_port, node_name))
                                not_all_output_shapes = True

                        if not_all_output_shapes:
                            raise Error('Not all output shapes were inferred or fully defined for node "{}". ' +
                                        refer_to_faq_msg(40),
                                        node_name)
                elif node.kind != 'data':
                    raise Error(
                        'There is no registered "infer" function for node "{}" with op = "{}". ' +
                        'Please implement this function in the extensions. ' +
                        refer_to_faq_msg(37),
                        node_name,
                        node.soft_get('op')
                    )
                node.is_partial_inferred = True
        except Exception as err:
            log.error('Cannot infer shapes or values for node "{}".'.format(node.soft_get('name')))
            log.error(str(err))
            log.error('')
            log.error('It can happen due to bug in custom shape infer function {}.'.format(node.soft_get('infer')))
            log.error('Or because the node inputs have incorrect values/shapes.')
            log.error('Or because input shapes are incorrect (embedded to the model or passed via --input_shape).')
            debug_messages = '\n'.join(
                ['Layer "' + node_name + '": ' + node_attrs['debug_message'] for node_name, node_attrs in
                 graph.nodes(data=True) if 'debug_message' in node_attrs])
            if debug_messages != "":
                log.error('')
                log.error('Other possible failure reasons are listed below:')
                log.error(debug_messages)
            if not debug_logger:
                log.error('Run Model Optimizer with --log_level=DEBUG for more information.')
            else:
                log.debug('Node "{}" attributes: {}'.format(node.soft_get('name'), node.graph.node[node.id]))
            raise Error('Stopped shape/value propagation at "{}" node. '.format(node.soft_get('name')) +
                        refer_to_faq_msg(38)) from err
        control_flow_infer(graph, n)


def override_batch(graph: Graph, batch: int):
    """
    Overrides batch for nodes with 'op' param set to 'Parameter'
    Parameters
    ----------
    graph: graph to operate on
    batch: user defined integer value to override batch
    """
    if batch is not None:
        in_nodes = graph.get_op_nodes(op='Parameter')
        for node in in_nodes:
            if not node.soft_get('fixed_batch', False):
                name = node.soft_get('name', node.id)
                idx, has_layout = get_dim_from_layout(node, 'N')
                if has_layout:
                    if idx is not None:
                        node['shape'][idx] = batch
                    else:
                        log.warning(
                            'Layout for input {} doesn\'t have batch dimension. Skipping this input.'.format(name))
                else:
                    validate_batch_in_shape(node['shape'], name)
                    node['shape'][0] = batch


def validate_batch_in_shape(shape, layer_name: str):
    """
    Raises Error #39 if shape is not valid for setting batch size
    Parameters
    ----------
    shape: current shape of layer under validation
    layer_name: name of layer under validation
    """
    if len(shape) == 0 or (shape[0] is not dynamic_dimension and shape[0] not in (-1, 0, 1)):
        raise Error(('The input layer {} has a shape {} defined in the model. \n\n' +
                     'When you use -b (--batch) option, Model Optimizer applies its value to the first ' +
                     'element of the shape if it is equal to -1, 0 or 1. Otherwise, this is the ambiguous ' +
                     'situation - Model Optimizer can not know in advance whether the layer has the batch ' +
                     'dimension or not.\n\n For example, you want to set batch dimension equals 100 ' +
                     'for the input layer "data" with shape (10,34). Although you can not use --batch, ' +
                     'you should pass --input_shape (100,34) instead of --batch 100. \n\n' +
                     'You can also tell Model Optimizer where batch dimension is located by specifying --layout. \n\n' +
                     refer_to_faq_msg(39))
                    .format(layer_name, shape))


def override_placeholder_shapes(graph: Graph, user_shapes: dict, batch=None):
    """
    This function overrides shapes for nodes with 'op' param set to 'Parameter' with shapes defined by users (only
    for inputs without in/out port specified).
    And override batch if batch was specified and shape for input is not None.
    :param graph: graph to operate on
    :param user_shapes: dictionary, that represents user defined nodes and shapes
    :param batch: user defined integer value to override batch
    """
    if user_shapes is None:
        # DON'T MOVE UPPER!!! WE NEED TO SET BATCH FIRST
        # user did not specify neither shapes nor inputs, keep models values
        return
    placeholders = graph.get_nodes_with_attributes(kind='op', op='Parameter')
    for node_id in placeholders:
        node_attrs = graph.node[node_id]
        shape = None
        if node_id in user_shapes:
            values = user_shapes[node_id]
            for value in values:
                if 'in' not in value and 'out' not in value:
                    shape = value['shape'] if value['shape'] is not None else None
                    break  # we assume only one specified shape for one input
        if shape is not None:
            node_attrs['shape'] = shape
        if batch is not None and node_attrs['shape'] is not None and len(node_attrs['shape']) > 0:
            node_attrs['shape'][0] = batch


def type_infer(graph: Graph):
    nodes = list(nx.topological_sort(graph))
    for n in nodes:
        node = Node(graph, n)
        if node.kind == 'op':
            node_name = node.soft_get('name')
            node_type_infer(node)
            log.debug('Type infer for node {}: {}'.format(node_name,
                                                          [port.get_data_type() for port in node.out_ports().values()]))
            """
            Save the precision of input ports in the nodes. It is not possible to get the precision after the port
            re-numbering because the port precision is defined for output port only and for input port it is determined
            with the output port producing data to the input port. When output port id is changed it is not possible to
            determine input port precision.
            """
            for out_port in node.out_ports().values():
                for dest_port in out_port.get_destinations():
                    if not dest_port.node.has_valid('_in_port_precision'):
                        dest_port.node['_in_port_precision'] = {}
                    dest_port.node['_in_port_precision'][dest_port.idx] = out_port.get_data_type()


def node_type_infer(node):
    if node.has_valid('type_infer'):
        node.type_infer(node)
    elif node.has_valid('data_type'):
        node.out_port(0).set_data_type(node.data_type)
    else:
        copy_type_infer(node)


def copy_type_infer(node):
    for out_port in node.out_ports().values():
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        if len(connected_in_ports) != 0:
            data_type = connected_in_ports[0].get_data_type()
            if data_type is not None:
                out_port.set_data_type(data_type)
            else:
                src_node = connected_in_ports[0].get_connection().get_source().node
                node_type_infer(src_node)
                out_port.set_data_type(connected_in_ports[0].get_data_type())
        else:
            raise Error('No input ports of node {} to determine data type'.format(node.soft_get('name')))


def reverse_infer(graph: Graph, nodes: list):
    nodes = reversed(nodes)
    debug_logger = log.getLogger().isEnabledFor(log.DEBUG)
    for n in nodes:
        node = Node(graph, n)
        if node.has_and_set('reverse_infer'):
            log.debug("Executed reverse infer for node '{}'".format(node.soft_get('name', node.id)))
            node.reverse_infer(node)

            if debug_logger:
                log.debug('-' * 20)
                log.debug('Reverse infer for {}'.format(node.soft_get('name')))
                log.debug('Op: {}'.format(node.soft_get('op')))
                log.debug('Outputs:')
                log_debug_dict(node.out_nodes(), 'outputs')

                log.debug('Inputs:')
                log_debug_dict(node.in_nodes(), 'inputs')

    parameters_with_no_shape = []
    for node in graph.get_op_nodes(op='Parameter'):
        if not node.has_valid('shape'):
            parameters_with_no_shape.append(node)

    if len(parameters_with_no_shape) == 0:
        return

    parameters_names = ''
    for idx, node in enumerate(parameters_with_no_shape):
        parameters_names += "'{}'".format(node.soft_get('name', node.id))
        if idx < len(parameters_with_no_shape) - 1:
            parameters_names += ', '

    if len(parameters_with_no_shape) > 0:
        raise Error("Model Optimizer is unable to deduce input shapes for the following Parameter nodes: {}. "
                    "Please use cli options --input or --input_shape to set model input shape.".format(parameters_names))
