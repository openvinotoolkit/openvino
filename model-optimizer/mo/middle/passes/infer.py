"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log

import networkx as nx
import numpy as np

# TODO remove it
from mo.graph.graph import Node, Graph
from mo.graph.graph import dict_includes
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg, shrink_str_value


def log_debug_dict(nodes_per_port: dict, direction_name: str):
    for port, node in nodes_per_port.items():
        value = shrink_str_value(node.soft_get('value'))
        log.debug('{}[{}]: shape = {}, value = {}'.format(direction_name, port, node.soft_get('shape'), value))


def is_fully_defined_shape(shape: np.ndarray):
    if -1 in shape:
        return False
    return True


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
    if not start_node is None:
        start_index = nodes.index(start_node)
        nx.set_node_attributes(G=graph.subgraph(nodes[start_index:]), name='is_partial_inferred', values=False)
    else:
        nx.set_node_attributes(G=graph, name='is_partial_inferred', values=False)
    debug_logger = log.getLogger().isEnabledFor(log.DEBUG)

    nx.set_node_attributes(G=graph, name='executable',
                           values={n: True for n in graph.get_nodes_with_attributes(kind='data')})

    for n in nodes:
        # Data Flow Infer
        try:
            node = Node(graph, n)
            node_name = node.soft_get('name')
            if node.has('is_partial_inferred') and not node.is_partial_inferred:
                if node.has('infer') and not node.infer is None:
                    if debug_logger:
                        log.debug('-' * 20)
                        log.debug('Partial infer for {}'.format(node.soft_get('name')))
                        log.debug('Op: {}'.format(node.soft_get('op')))
                    node.infer(node)
                    out_nodes = node.out_nodes()

                    # propagate nchw_layout attributes to data nodes
                    if node.has('nchw_layout'):
                        for out_node in out_nodes.values():
                            out_node['nchw_layout'] = node.nchw_layout

                    # In debug print current node attributes, input shapes/values and output shape/values
                    if debug_logger:
                        log.debug('Inputs:')
                        log_debug_dict(node.in_nodes(), 'input')
                        log.debug('Outputs:')
                        log_debug_dict(node.out_nodes(), 'output')

                    not_all_output_shapes = False

                    for out_port, out_node in out_nodes.items():
                        not_all_output_shapes = False
                        if not out_node.has_valid('shape'):
                            log.error('Shape is not defined for output {} of "{}".'.format(out_port, node_name))
                            not_all_output_shapes = True
                        elif not is_fully_defined_shape(out_node.shape):
                            log.error(
                                ('Shape {} is not fully defined for output {} of "{}". ' +
                                 'Use --input_shape with positive integers to override model input shapes.').format(
                                    out_node.shape,
                                    out_port,
                                    node_name
                                )
                            )
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

    not_fully_inferred = graph.get_nodes_with_attributes(is_not_fully_inferred=True)
    for n in not_fully_inferred:
        node = Node(graph, n)
        if node.has('infer') and not node.infer is None:
            node.infer(node)

    return graph


def override_batch(graph: Graph, batch: int):
    """
    Overrides batch for nodes with 'op' param set to 'Parameter'
    Parameters
    ----------
    graph: graph to operate on
    batch: user defined integer value to override batch
    """
    if batch is not None:
        for node_id, data in graph.nodes(data=True):
            if 'op' in data and data['op'] == 'Parameter' and not data.get('fixed_batch', False):
                if len(data['shape']) == 0 or data['shape'][0] not in (-1, 0, 1):
                    raise Error(('The input layer {} has a shape {} defined in the model. \n\n' +
                                 'When you use -b (--batch) option, Model Optimizer applies its value to the first ' +
                                 'element of the shape if it is equal to -1, 0 or 1. Otherwise, this is the ambiguous ' +
                                 'situation - Model Optimizer can not know in advance whether the layer has the batch ' +
                                 'dimension or not.\n\n For example, you want to set batch dimension equals 100 ' +
                                 'for the input layer "data" with shape (10,34). Although you can not use --batch, ' +
                                 'you should pass --input_shape (100,34) instead of --batch 100. \n\n' +
                                 refer_to_faq_msg(39))
                                .format(data['name'], data['shape']))
                data['shape'][0] = batch


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


def update_fully_connected_shapes(graph: Graph):
    nodes = nx.topological_sort(graph)
    while True:
        should_infer = False
        for n in nodes:
            node = Node(graph, n)
            if node.has('type') and node.type == 'MatMul' and node.in_node(0).shape.size == 3:
                log.debug("node.in_node(0).shape = {}".format(node.in_node(0).shape))
                log.debug("channel_dims = {}".format(node.channel_dims))
                assert (node.in_node(0).shape.size == 3 and node.channel_dims > 0)
                node.in_node(0).shape = np.delete(node.in_node(0).shape, 1)
                if node.out_node().shape.size == 3:
                    node.channel_dims = node.channel_dims - 1
                    log.debug("Initiated partial infer from update_fully_connected_shapes")
                    graph = partial_infer(graph, node.in_node(0).id)
                    # Not working
                    # graph = mark_dead_nodes(graph)
                    # graph = eliminate_dead_nodes(graph)
                    should_infer = True
                    break
        if not should_infer:
            break

    if graph.graph['cmd_params'].generate_experimental_IR_V10:
        return