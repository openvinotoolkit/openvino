"""
 Copyright (c) 2018 Intel Corporation

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
from mo.front.extractor import update_ie_fields
from mo.graph.graph import Node, get_outputs, get_node_id_by_name
from mo.middle.passes.eliminate import get_nodes_with_attributes
from mo.middle.pattern_match import apply_pattern, for_each_sub_graph
from mo.ops.lin_op import Mul, Add
from mo.ops.op import Op
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def log_debug_dict(nodes_per_port: dict, direction_name: str):
    for port, node in nodes_per_port.items():
        value = str(node.soft_get('value'))
        max_symbols = 100
        if len(value) > max_symbols:
            value = value.strip('\n')[:max_symbols - 3] + '...'
        log.debug('{}[{}]: shape = {}, value = {}'.format(direction_name, port, node.soft_get('shape'), value))


def is_fully_defined_shape(shape: np.ndarray):
    if -1 in shape:
        return False
    return True


def control_flow_infer(graph: nx.MultiDiGraph, node_name: str):
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
    is_executable_df = not all([not graph.node[u]['executable'] for u, _, attrs in in_df_edges_with_data]
                               if len(in_df_edges_with_data) else [False])
    is_executable_cf = not any([not graph.node[u]['executable'] for u, _, attrs in in_cf_edges_with_data]
                               if len(in_cf_edges_with_data) else [False])
    is_executable = is_executable_df and is_executable_cf

    node = Node(graph, node_name)
    if 'cf_infer' in graph.node[node_name] and callable(node.cf_infer):
        node.cf_infer(node, is_executable, mark_executability)
    else:
        for _, out_data in graph.out_edges(node_name):
            mark_executability(out_data, is_executable)


def delete_not_executable(graph: nx.MultiDiGraph):
    nodes_to_remove = set()
    for node_name, node_attrs in graph.nodes(data=True):
        if node_attrs['kind'] == 'data' and 'executable' in node_attrs and not node_attrs['executable']:
            [nodes_to_remove.add(op) for op, _ in graph.in_edges(node_name)]
            nodes_to_remove.add(node_name)
    log.debug('Removing the following not executable nodes: {}'.format('\n'.join(sorted(map(str, nodes_to_remove)))))
    graph.remove_nodes_from(nodes_to_remove)


def delete_control_flow_edges(graph: nx.MultiDiGraph):
    for u, v, k, attrs in list(graph.edges(keys=True, data=True)):
        if 'control_flow_edge' in attrs and attrs['control_flow_edge']:
            graph.remove_edge(u, v, k)
            log.debug('Removing control flow edge from {} to {}'.format(u, v))


def partial_infer(graph: nx.MultiDiGraph, start_node: str = None):
    """
    Tries to execute constant parts of the graph and deduce as much as possible
    information following the data flow, e.g. calculate and propagate shapes and
    constant values. Partially or completely defined values are stored in data
    nodes (kind='data').
    """
    cycle_nodes = get_nodes_with_attributes(graph, is_cyclic=True)
    cycle_nodes = [Node(graph, node).out_node().id for node in cycle_nodes]
    ebunch = list(graph.out_edges(nbunch=cycle_nodes, data=True, keys=True))
    graph.remove_edges_from(ebunch)

    try:
        nodes = list(nx.topological_sort(graph))
    except:
        raise Error('Graph contains a cycle. Can not proceed. ' + refer_to_faq_msg(97))

    graph.add_edges_from(ebunch)

    # Mark all nodes as not inferred yet
    if not start_node is None:
        start_index = nodes.index(start_node)
        nx.set_node_attributes(graph.subgraph(nodes[start_index:]), name='is_partial_inferred', values=False)
    else:
        nx.set_node_attributes(graph, name='is_partial_inferred', values=False)
    debug_logger = log.getLogger().isEnabledFor(log.DEBUG)

    nx.set_node_attributes(graph, name='executable',
                           values={n: True for n in get_nodes_with_attributes(graph, kind='data')})

    for n in nodes:
        # Data Flow Infer
        try:
            node = Node(graph, n)
            node_name = node.soft_get('name')
            if node.has('is_partial_inferred') and not node.is_partial_inferred:
                if node.has('infer') and not node.infer is None:
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

    not_fully_inferred = get_nodes_with_attributes(graph, is_not_fully_inferred=True)
    for n in not_fully_inferred:
        node = Node(graph, n)
        if node.has('infer') and not node.infer is None:
            node.infer(node)

    #delete_not_executable(graph)
    return graph


def check_for_cycle(graph: nx.MultiDiGraph):
    is_acyclic = nx.is_directed_acyclic_graph(graph)
    if not is_acyclic:
        raise Error('Graph contains a cycle. Can not proceed. ' + refer_to_faq_msg(97))


def mark_outputs(graph: nx.MultiDiGraph):
    nx.set_node_attributes(graph, name='is_output', values=False)
    for node in graph.nodes():
        if graph.node[node]['kind'] == 'data' and len(get_outputs(graph, node)) == 0:
            nx.set_node_attributes(graph, name='is_output', values={node: True})


def override_batch(graph: nx.MultiDiGraph, batch: int):
    """
    Overrides batch for nodes with 'op' param set to 'Placeholder'
    Parameters
    ----------
    graph: graph to operate on
    batch: user defined integer value to override batch
    """
    if batch is not None:
        for node_id, data in graph.nodes(data=True):
            if 'op' in data and data['op'] == 'Placeholder':
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


def override_placeholder_shapes(graph: nx.MultiDiGraph, user_shapes: dict, batch=None):
    """
    Overrides shapes for nodes defined by user
    Or overrides shapes for nodes with 'op' param set to 'Placeholder'
    Parameters
    ----------
    graph: graph to operate on
    user_shapes: dictionary, that represents user defined nodes and shapes
    batch: user defined integer value to override batch
    """
    if user_shapes is None:
        # DON'T MOVE UPPER!!! WE NEED TO SET BATCH FIRST
        # user did not specify neither shapes nor inputs, keep models values
        return
    if isinstance(user_shapes, dict):
        for node_id, values in user_shapes.items():
            for value in values:
                shape = value['shape'] if 'shape' in value else None
                if shape is not None:
                    graph.node[node_id]['shape'] = shape
                if 'shape' in graph.node[node_id] and graph.node[node_id]['shape'] is not None:
                    if batch:
                        old_batch = graph.node[node_id]['shape'][0]
                        if old_batch != batch:
                            graph.node[node_id]['shape'] = np.array([batch, *graph.node[node_id]['shape'][1:]])


def _scale_input_action_mul(graph: nx.MultiDiGraph, match: dict, scale: float):
    assert (len(match['placeholder'].out_nodes()))

    tinput = match['placeholder']
    if not tinput.has_valid('shape'):
        raise Error("Node {} has not valid shape attribute".format(tinput.id))

    input_shape = tinput.shape
    toutput = match['data']

    # Create Mul node
    value = np.array([1 / scale])

    # Disconnect input with data node
    graph.remove_edge(tinput.id, toutput.id)

    # Create Mul node
    mul_node = Mul(graph, dict(name="Mul1_"))
    mul_data = Op.create_input_data_node(graph, "data_mul_scale_", np.array(value))
    Op.expand_node_shape(mul_data, len(input_shape) - 2 if graph.graph['layout'] == 'NCHW' else 0)
    mul_input = Op.create_data_node(graph, tinput, {'shape': toutput.shape})

    mul_node.create_node_with_data(inputs=[mul_input, mul_data], data_nodes=toutput)


def scale_input(graph: nx.MultiDiGraph, scale: float):
    """
    Searches for all entries of Placeholder in graph and passes it to the the replace transform
    Args:
        graph: an instance of nx graph
        scale: integer value for the scale
    """
    if scale is None or scale == 1:
        return

    apply_pattern(
        graph,
        nodes=[
            ('placeholder', dict(kind='op', op='Placeholder')),
            ('data', dict(kind='data'))],
        edges=[
            ('placeholder', 'data'), ],
        action=lambda graph, match: _scale_input_action_mul(graph, match, scale)
    )


def add_mean_scale_values(graph: nx.MultiDiGraph, values):
    input_nodes = {}
    for node in graph.nodes():
        node = Node(graph, node)
        if node.has_valid('op') and node.op == 'Placeholder':
            input_nodes.update({node.id: node})

    if not isinstance(values, dict):
        if len(values) != len(input_nodes):
            raise Error('Numbers of inputs and mean/scale values do not match. ' +
                        refer_to_faq_msg(61))

        data = np.copy(values)
        values = {}
        for idx, key in enumerate(input_nodes.keys()):
            values.update(
                {
                    input_nodes[key]['name']: {
                        'mean': data[idx][0],
                        'scale': data[idx][1]
                    }
                }
            )

    for node_name in values:
        node_id = get_node_id_by_name(graph, node_name)
        node_mean_scale_values = values[node_name]
        if node_id not in input_nodes:
            # if the user cutted-off input of the network then input node name specified in the --scale_values
            # or --mean_values doesn't correspond to a real input node generated by Model Optimizer. But the information
            # about initial input node name is stored in Placeholder's attribute 'initial_node_name'
            new_node_id = None
            for placeholder in input_nodes.values():
                if placeholder.has('initial_node_name') and placeholder.initial_node_name == node_name:
                    new_node_id = placeholder.id
                    break
            if new_node_id is None:
                raise Error('Input with name {} wasn\'t found!'.format(node_name) +
                            refer_to_faq_msg(83))
            node_id = new_node_id

        input_node = Node(graph, node_id)
        apply_scale(graph, input_node, node_mean_scale_values)
        apply_mean_value(graph, input_node, node_mean_scale_values)


def apply_scale(graph: nx.MultiDiGraph, input_node: Node, node_mean_scale_values: dict):
    if 'scale' in node_mean_scale_values and node_mean_scale_values['scale'] is not None:
        if all([x == 1 for x in node_mean_scale_values['scale']]):
            return
        out_node = input_node.out_node()
        if not input_node.has_valid('shape'):
            raise Error("Node {} has not valid shape attribute".format(input_node.id))
        input_shape = input_node.shape

        # Create Mul node
        value = 1 / np.array(node_mean_scale_values['scale'])
        graph.remove_edge(input_node.id, out_node.id)

        mul_node = Mul(graph, dict(name="Mul_"))
        mul_data = Op.create_input_data_node(graph, "data_mul_", np.array(value))
        Op.expand_node_shape(mul_data, (len(input_shape) - 2 if graph.graph['layout'] == 'NCHW' else 0))
        mul_input = Op.create_data_node(graph, input_node, {'shape': out_node.shape})

        mul_node.create_node_with_data(inputs=[mul_input, mul_data], data_nodes=out_node)


def apply_mean_value(graph: nx.MultiDiGraph, input_node: Node, node_mean_scale_values: dict):
    if 'mean' in node_mean_scale_values and node_mean_scale_values['mean'] is not None:
        if all([x == 0 for x in node_mean_scale_values['mean']]):
            return
        out_node = input_node.out_node()
        if not input_node.has_valid('shape'):
            raise Error("Node {} has not valid shape attribute".format(input_node.id))
        input_shape = input_node.shape
        # Create Add node
        graph.remove_edge(input_node.id, out_node.id)

        value = np.array(node_mean_scale_values['mean']) * (-1)

        add_node = Add(graph, dict(name="Add_"))
        add_data = Op.create_input_data_node(graph, "data_add_", np.array(value))
        Op.expand_node_shape(add_data, (len(input_shape) - 2 if graph.graph['layout'] == 'NCHW' else 0))
        add_input = Op.create_data_node(graph, input_node, {'shape': out_node.shape})

        add_node.create_node_with_data(inputs=[add_input, add_data], data_nodes=out_node)


def update_fully_connected_shapes(graph: nx.MultiDiGraph):
    nodes = nx.topological_sort(graph)
    while True:
        should_infer = False
        for n in nodes:
            node = Node(graph, n)
            if node.has('type') and node.type == 'FullyConnected' and node.in_node(0).shape.size == 3:
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


# Convert MUL operation to Power layer in case when
# mul op takes two inputs (scalar constant and tensor)
def convert_mul_add_to_power(graph: nx.MultiDiGraph):
    for_each_sub_graph(graph, convert_mul_add_to_power)
    nodes = list(graph.nodes())
    for n in nodes:
        # As we remove nodes from graph, we should check that node exists in graph
        if n in graph:
            node = Node(graph, n)
            if node.has('op') and (node.op == 'Mul' or node.op == 'Add') and len(node.in_nodes()) == 2 and \
                    node.soft_get('can_be_scaleshift') is not False:
                scalar_idx, tensor_idx = (0, 1) if not node.in_node(0).value is None else (1, 0)
                if not node.in_node(scalar_idx).value is None and node.in_node(tensor_idx).value is None:
                    if np.squeeze(node.in_node(scalar_idx).value).ndim == 0:
                        node['type'] = 'Power'
                        node['scale'] = node.in_node(scalar_idx).value.item() if node.op == 'Mul' else 1
                        node['power'] = 1
                        node['shift'] = node.in_node(scalar_idx).value.item() if node.op == 'Add' else 0
                        node['op'] = 'Power'
                        if node.has('operation'):
                            del node.graph.node[node.id]['operation']
                        update_ie_fields(graph.node[node.id])
                        scalar_node = node.in_node(scalar_idx)
                        graph.remove_edge(scalar_node.id, node.id)
                        graph.remove_node(scalar_node.id)
