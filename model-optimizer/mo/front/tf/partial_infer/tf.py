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
from re import match

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from mo.front.extractor import node_defs_to_str
from mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape, get_tf_node_port
from mo.graph.graph import Node
from mo.utils.graph import node_incoming_neighbourhood, node_outcoming_neighbourhood


def tf_native_tf_node_infer(node: Node):
    """
    The infer function should be used to infer shape and data type of the TF operation not supported by IE.
    :param node: node to infer.
    :return: None
    """
    log.info('Called "tf_native_tf_node_infer" for node "{}"'.format(node.id))

    # create a sub-graph only to make inference. The sub-graph contains desired node and it's inputs neighbourhood of
    # depth 10. The number 10 is quite big to be sure that determine_data_type function will be able to identify the
    # data type of input tensors, but not too huge to contain the whole graph.
    # Also the sub-graph contains names of the output nodes of the node to perform native infer.
    nodes_to_extract = node_incoming_neighbourhood(node.graph, node.id, 10) + node_outcoming_neighbourhood(node.graph,
                                                                                                           node.id, 1)
    tmp_graph = node.graph.create_sub_graph_copy(nodes_to_extract)

    tmp_node_attrs = tmp_graph.node[node.id]
    tmp_node = Node(tmp_graph, node.id)

    # node attributes that are required by 'infer_subgraph_output_nodes' function
    lists_to_init = ['input_nodes_names', 'output_tensors_names', 'nodes_order', 'internal_output_node_name',
                     'real_input_dims']

    for item in lists_to_init:
        tmp_node_attrs[item] = list()
    tmp_node_attrs['pbs'] = {tmp_node.name: tmp_node.pb}
    tmp_node_attrs['nodes_order'].append(tmp_node.id)
    for ind in range(len(tmp_node.out_edges())):
        tmp_node_attrs['output_tensors_names'].append(tmp_node.id + ":" + str(ind))

    tf_subgraph_infer(tmp_node)
    # the shape and value has been inferred and saved to the tmp_node's out nodes attribute. Let's copy it back!
    for tmp_out_port, tmp_out_node in tmp_node.out_nodes().items():
        if tmp_out_node.value is not None:
            node.out_node(tmp_out_port).value = np.array(tmp_out_node.value)
        if tmp_out_node.shape is not None:
            node.out_node(tmp_out_port).shape = np.array(tmp_out_node.shape)
        if tmp_out_node.data_type is not None:
            node.out_node(tmp_out_port).data_type = tmp_out_node.data_type
    # lets cleanup the temporary graph
    tmp_graph.clear()


def generate_feed_dict(graph: tf.Graph, node: Node):
    """
    The first value in the return tuple is True if all inputs for the node has constant values.
    The second returned value is mapping of placeholder tensor to the numpy arrays with the values for these
    placeholders.
    :param graph: the TensorFlow Graph to generate feed dictionary to.
    :param node: the node which represents TensorFlow sub-graph of operations.
    :return: pair where the first element is a flag that specifies that all node inputs are constants and a dictionary
    where key is the input Tensor object and the value is the tensor value.
    """
    all_constants = True
    feed_dict = dict()
    for in_data_node_name, edge_attrs in node.get_inputs():
        if 'control_flow_edge' in edge_attrs and edge_attrs['control_flow_edge']:
            continue
        value = node.in_node(edge_attrs['in']).value
        if value is None:
            all_constants = False
            placeholder_pb = node['pbs'][edge_attrs['placeholder_name']]
            value = np.ones(shape=tf_tensor_shape(placeholder_pb.attr['shape'].shape),
                            dtype=tf_dtype_extractor(placeholder_pb.attr['dtype'].type))
        feed_dict[graph.get_tensor_by_name(edge_attrs['placeholder_name'] + ":0")] = value
    return all_constants, feed_dict


def get_subgraph_output_tensors(node: Node):
    """
    Infer output shapes of the node. The function uses TF to infer the values of output tensors and then getting tensor
    shape.
    TODO: try to not infer values but just infer the output tensors shapes.
    :param node: sub-graph node to infer.
    :return: pair where the first element is a flag that specifies that all node inputs are constants and a dictionary
    where key is the output port and the value is the tensor value.
    """
    result = dict()
    graph_def = tf.GraphDef()
    text_format.Merge(node_defs_to_str(node), graph_def)
    tf.reset_default_graph()
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():  # pylint: disable=not-context-manager
        with sess.as_default():  # pylint: disable=not-context-manager
            tf.import_graph_def(graph_def, name='')
            all_constants, feed_dict = generate_feed_dict(graph, node)
            for out_port, out_tensor_name in enumerate(node['output_tensors_names']):
                if not match('.*:\d+', out_tensor_name):
                    out_tensor_name = out_tensor_name + ":" + str(out_port)
                result_tensor = sess.run(graph.get_tensor_by_name(out_tensor_name), feed_dict=feed_dict)
                result[out_port] = result_tensor
    return all_constants, result


def tf_subgraph_infer(node: Node):
    """
    Infer output shapes of the node using TF to infer the values of output tensors and then getting tensor shapes.
    If all inputs of the node are constants then the node's attribute 'value' is updated also.
    :param node: sub-graph node to infer. The function updates 'shape' and 'data_type' attributes of the node.
    :return: None
    """
    # TODO: try to not infer values but just infer the output tensors shapes.
    add_placeholders_to_subgraph(node)

    all_constants, output_tensors = get_subgraph_output_tensors(node)
    for out_port, tensor_value in output_tensors.items():
        out_node = node.out_node(out_port)
        out_node.shape = np.array([dim for dim in tensor_value.shape])
        out_node.data_type = tensor_value.dtype
        log.debug("Inferred shape of the output tensor with index '{}' of the node '{}': '{}'".format(str(out_port),
                                                                                                      node.name,
                                                                                                      out_node.shape))
        if all_constants:
            out_node.value = tensor_value


def add_node_def_to_subgraph(subgraph_node: Node, node_def: tf.NodeDef, name: str = None, position: int = 0,
                             is_input: bool = False):
    """
    Adds NodeDef definition of the node to the internal structures of the sub-graph's_node object that represents a
    sub-graph of operations.
    :param subgraph_node: the node that represents sub-graph where new node should be added.
    :param node_def: the NodeDef (TF operation, variable or constant) to be added to the sub-graph.
    :param name: name how to save added node. Default value is None which means take name from the NodeDef.
    :param position: position in the GraphDef where to put the NodeDef. Default value is 0.
    :param is_input: flag that specifies whether the node is input for the sub-graph. Default value is False.
    :return: None
    """
    name = name or node_def.name
    assert (name not in subgraph_node['pbs'].keys())
    if is_input:
        subgraph_node['input_nodes_names'].append(name)
    subgraph_node['pbs'][node_def.name] = node_def
    subgraph_node['nodes_order'].insert(position, name)


def determine_data_type(node: Node):
    """
    Tries to determine data type of the node. The input node could be either data or op node. If we don't know the data
    type of the node then we recursively check the first parent of the node.
    :param node: node to determine data type.
    :return: data type of the node output in the numpy format.
    """
    if node.has_and_set('data_type'):
        return node.data_type
    if node.has_and_set('kind') and node.kind == 'op':
        if node.has_and_set('pb'):
            if 'dtype' in node.pb.attr:
                return tf_dtype_extractor(node.pb.attr['dtype'].type)
            if 'T' in node.pb.attr:
                return tf_dtype_extractor(node.pb.attr['T'].type)
    if node.has_and_set('kind') and node.kind == 'data':
        if 'value' in node and node.value is not None:
            return node.value.dtype
    if len(node.in_nodes()) != 0:  # try to guess data type from the first parent
        return determine_data_type(node.in_node(0))
    log.error('Failed to determine data type for node "{}"'.format(node.name))
    return None


def add_placeholders_to_subgraph(node: Node):
    """
    Adds placeholders to the node's list of protobufs based on input nodes to the subgraph (the value of
    'internal_input_node_name' property).
    The function also updates input tensors for nodes which consume output of nodes that were replaced with
    placeholders.
    :param node: the node to add placeholders to.
    :return: None
    """
    inputs_replacements = list()
    for index, (in_data_node, edge_attrs) in enumerate(node.get_sorted_inputs()):
        if 'control_flow_edge' in edge_attrs and edge_attrs['control_flow_edge']:
            continue

        if 'internal_input_node_name' in edge_attrs.keys():
            input_tensor_name = edge_attrs['internal_input_node_name']
        else:
            input_tensor_name = node['pb'].input[index]

        input_node_name, port = get_tf_node_port(input_tensor_name)

        placeholder_name = placeholder_name_for_node(input_node_name, port)
        edge_attrs['placeholder_name'] = placeholder_name
        in_node = node.in_node(index)

        assert in_node.shape is not None

        if placeholder_name not in node['pbs'].keys():
            placeholder = tf.placeholder(determine_data_type(in_node), in_node.shape, placeholder_name)
            inputs_replacements.append((input_tensor_name, placeholder_name))
            add_node_def_to_subgraph(node, placeholder.op.node_def, is_input=True)
            log.debug("Added placeholder with name '{}'".format(placeholder_name))

    # update initial input names to a transposed ones
    for old_input_tensor_name, new_name in inputs_replacements:
        update_input_in_pbs(node, old_input_tensor_name, new_name)


def update_input_in_pbs(node: Node, old_input_tensor_name: str, new_input_name: str):
    """
    The function replaces all inputs with name 'old_input_tensor_name' with a
    new input with name 'new_input_name'. This transformation is applied
    for all NodeDef objects in the 'pbs' list.
    """
    log.debug("update_input_in_pbs: replace input '%s' with input '%s'" % (old_input_tensor_name, new_input_name))
    old_input_tensor_name_without_port = old_input_tensor_name.split(":")[0]
    for pb in node['pbs'].values():
        if hasattr(pb, 'input'):
            for ind in range(len(pb.input)):
                if pb.input[ind] == old_input_tensor_name or pb.input[ind] == old_input_tensor_name_without_port:
                    pb.input[ind] = new_input_name
                    log.debug("Replacing input '{}' of the node '{}' with placeholder '{}'".format(ind, pb.name,
                                                                                                   new_input_name))


def placeholder_name_for_node(node_name: str, output_port: int):
    return node_name + "_port_" + str(output_port) + "_ie_placeholder"
