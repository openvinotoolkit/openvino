"""
 Copyright (C) 2018-2020 Intel Corporation

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
import os

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.ReduceOps import ReduceOp
from extensions.ops.activation_ops import Activation
from extensions.ops.elementwise import Elementwise, LogicalElementwise, BiasAdd, Div, Mul, Pow, Sub
from extensions.ops.psroipooling import DeformablePSROIPoolingOp
from extensions.ops.scatter import Scatter
from extensions.ops.split import Split, VariadicSplit
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.ops.clamp import AttributedClamp
from mo.ops.convolution import Convolution
from mo.ops.deconvolution import Deconvolution
from mo.ops.op import Op
from mo.ops.pooling import Pooling
from mo.ops.result import Result
from mo.utils.class_registration import update_registration
from mo.utils.import_extensions import import_by_path
from mo.utils.ir_reader.extender import Extender

from extensions.back.TopKNormalizer import TopKNormalizer

# Operations not registred in collect_ops() function
custom_ops = {
    'AvgPool': Pooling,
    'BiasAdd': BiasAdd,
    'Convert': Cast,
    'ConvolutionBackpropData': Deconvolution,
    'DeformablePSROIPooling': DeformablePSROIPoolingOp,
    'Divide': Div,
    'GroupConvolution': Convolution,
    'GroupConvolutionBackpropData': Deconvolution,
    'MaxPool': Pooling,
    'Multiply': Mul,
    'Power': Pow,
    'Split': Split,
    'Subtract': Sub,
    'VariadicSplit': VariadicSplit,
    'Clamp': AttributedClamp,
}


def collect_ops(path: str):
    """
    A function to registrate all MO ops
    :param path: Path to Model Optimizer folder
    :return:
    """
    import_by_path(os.path.join(path, 'mo', 'ops'), ['mo', 'ops'])
    import_by_path(os.path.join(path, 'extensions', 'ops'), ['extensions', 'ops'])
    update_registration(classes=[Op, Activation, Elementwise, LogicalElementwise, ReduceOp, Scatter],
                        enabled_transforms=[], disabled_transforms=[])


def collect_extenders(path: str):
    """
    A function to registrate all MO IR Reader extenders
    :param path: Path to Model Optimizer folder
    :return:
    """
    import_by_path(os.path.join(path, 'mo', 'utils', 'ir_reader', 'extenders'),
                   ['mo', 'utils', 'ir_reader', 'extenders'])
    update_registration(classes=[Extender], enabled_transforms=[], disabled_transforms=[])


def collect_node_outputs(node: Node) -> dict:
    """
    Function to collects output connections of node.
    :param node: node to collect connections
    :return: dictionary of the form {out_port: [(input_port, destination_node_id)]}
    """
    result = dict()
    for out_port_idx, out_port in node.out_ports().items():
        dest_info = []
        for d in out_port.get_destinations():
            dest_info.append((d.idx, d.node.id))
        result[out_port_idx] = dest_info
    return result


def restore_correct_ports(graph: Graph):
    """
    Function renumbers from IE to MO port numbering and add ports to all nodes in graph.
    :param graph:
    :return:
    """
    for node_id, attrs in graph.nodes(data=True):
        if '_in_ports' not in attrs:
            attrs['_in_ports'] = set()
        if '_out_ports' not in attrs:
            attrs['_out_ports'] = set()

    for u, v, k, d in graph.edges(data=True, keys=True):
        from_node_attrs = graph.node[u]
        to_node_attrs = graph.node[v]
        is_control_flow = 'control_flow_edge' in d and d['control_flow_edge'] is True

        if 'in' in d:
            in_port_id = d['in'] if not is_control_flow else 'control_flow_' + str(d['in'])
            to_node_attrs['_in_ports'].update({in_port_id: {'control_flow': is_control_flow}})
        if 'out' in d:
            num_of_in_nodes = len(Node(graph, u).in_nodes())
            decremented_number = d['out'] - num_of_in_nodes
            out_port_id = decremented_number if not is_control_flow else 'control_flow_' + str(decremented_number)
            from_node_attrs['_out_ports'].update({out_port_id: {'control_flow': is_control_flow}})
            d['out'] = decremented_number


def propagate_const_values(op: Node):
    """
    Function propagates const value from input data node and reshape it to correct shape.
    :param op:
    :return:
    """
    assert op.soft_get('type') == 'Const', 'Wrong operation type, {} instead of Const!' \
                                           ''.format(op.soft_get('type'))

    in_data_node = op.in_node()
    out_data_node = op.out_node()

    value = in_data_node.value
    assert len(op.out_node(0).out_nodes()) > 0, 'Const node {} have no consumers.'.format(op.soft_get('name'))
    if op.out_node(0).out_node(0).type == 'BinaryConvolution':
        # Unpack binary weights for binary convolution (revert PackBinaryWeights transformation)
        weights_rounded = np.unpackbits(value)
        weights_rounded.dtype = np.int8
        for elem in range(len(weights_rounded)):
            if weights_rounded[elem] == 0:
                weights_rounded[elem] -= 1  # pylint: disable=unsupported-assignment-operation
        assert len(weights_rounded) % 8 == 0
        weights_rounded = weights_rounded.reshape([len(weights_rounded) // 8, 8])  # pylint: disable=no-member
        weights_rounded = np.flip(weights_rounded, axis=1)
        value = weights_rounded.flatten()

    op['shape'] = out_data_node.shape
    # Reshape data node value for correct shape
    op['value'] = np.reshape(value, op.shape)


def groupconv_to_conv(op: Node):
    """
    Function makes GroupConv op back to Conv op with weights reshaping
    :param op:
    :return:
    """
    assert op.soft_get('type') == 'GroupConvolution', \
        'Wrong operation type, {} instead of GroupConvolution!'.format(op.soft_get('type'))

    weights_shape = op.in_port(1).data.get_shape()
    group = weights_shape[0]
    new_shape = [weights_shape[1] * group, *weights_shape[2:]]

    weights_node = op.in_port(1).get_source().node
    if weights_node.type == 'Const':
        weights_node.value = np.reshape(weights_node.value, new_shape)
    elif weights_node.type == 'Reshape':
        # we remove reshape node added in ConvolutionWithGroupsResolver pass
        assert weights_node.in_port(0).get_source().data.get_shape() == new_shape, \
            'Weight shape and calculated shape mismatch in GroupConv node {}.'.format(op.name)
        op.in_port(1).disconnect()
        weights_node.in_port(0).get_source().get_connection().set_destination(op.in_port(1))
    else:
        assert op.in_port(1).get_source().data.get_shape() == new_shape, \
            'Weight shape and calculated shape mismatch in GroupConv node {}.'.format(op.name)
    # we need to set this attrs for correct shape infer as convolution
    op['group'] = group
    op.type = 'Convolution'


def backprop_to_deconv(op: Node):
    """
    Function changes BackpropData operations type to correct creation
    :param op:
    :return:
    """
    assert op.soft_get('type') in ('ConvolutionBackpropData', 'GroupConvolutionBackpropData'), \
        'Wrong operation type, {} instead of ConvolutionBackpropData/GroupConvolutionBackpropData!' \
        ''.format(op.soft_get('type'))

    if op.has_valid('output_padding'):
        # In this case we need to create Deconvolution as Convolution
        op['type_to_create'] = 'Convolution'
    op['old_input_shapes'] = list()
    for n in op.in_nodes():
        op.old_input_shapes.append(int64_array(op.in_node(n).shape))


def ti_add_edge_attrs(op: Node):
    """
    Function adds necessary edge attrs in TensorIterator node
    :param op:
    :return:
    """
    assert op.soft_get('type') == 'TensorIterator', 'Wrong operation type, {} instead of TensorIterator!' \
                                                    ''.format(op.soft_get('type'))

    i = 0
    for num in range(len(op.in_ports())):
        op.in_port(num).external_port_id = i
        i += 1
    for num in range(len(op.out_ports())):
        op.out_port(num).external_port_id = i
        i += 1


def assign_add_output_result(op: Node):
    """
    Function adds necessary output result node for Assign node
    :param op:
    :return:
    """
    assert op.soft_get('type') == 'Assign', 'Wrong operation type, {} instead of Assign!' \
                                            ''.format(op.soft_get('type'))
    tmp_result = Result(op.graph, {'name': op.soft_get('name', op.id) + '/Result'}).create_node()
    op.out_port(0).connect(tmp_result.in_port(0))


def topk_add_output_result(op: Node):
    """
    Function adds missed output result nodes for TopK node
    :param op:
    :return:
    """
    assert op.soft_get('type') == 'TopK', 'Wrong operation type, {} instead of TopK!'.format(op.soft_get('type'))

    TopKNormalizer.normalize_outputs(op)


def copy_input_blobs(op: Node, copy_op: Node):
    """
    Function copy input blob data nodes from restored graph to copied one
    :param op: Node from restored graph
    :param copy_op: Node from copied graph
    :return:
    """
    for u, d in op.get_sorted_inputs():
        if 'bin' in d:
            Op.create_and_connect_input_data_node(copy_op.graph, copy_op,
                                                  {'value': op.in_node(d['in']).value,
                                                   'shape': op.in_node(d['in']).shape}, d)


# Map with preprocessing functions
preprocessing_op_nodes = {
    'Const': propagate_const_values,
    'GroupConvolution': groupconv_to_conv,
    'ConvolutionBackpropData': backprop_to_deconv,
    'GroupConvolutionBackpropData': backprop_to_deconv,

}

# Map with postprocessing functions for nodes
postprocessing_op_nodes = {
    'Assign': assign_add_output_result,
    'TensorIterator': ti_add_edge_attrs,
    'TopK': topk_add_output_result,
}


def copy_graph_with_ops(graph: Graph) -> Graph:
    """
    Function to copy graph and apply extenders to appropriate nodes
    :param graph: Graph to copy
    :return:Copied graph with applyed extenders
    """
    new_graph = Graph()
    new_graph.stage = 'back'
    new_graph.graph = graph.graph

    node_connections = dict()
    mapping_of_old_idx_into_new = dict()

    restore_correct_ports(graph)

    # Nodes preprocessing stage in source graph
    # Firstly propagate values only for Const nodes, because other preprocessings
    # assumes Const nodes are already preprocessed.
    for op in graph.get_op_nodes(type='Const'):
        preprocessing_op_nodes[op.type](op)

    for op in graph.get_op_nodes():
        if op.soft_get('type') != 'Const' and op.soft_get('type') in preprocessing_op_nodes:
            preprocessing_op_nodes[op.type](op)

    # Create a new copy of graph with correct attributes (shape & type infer, backend attrs etc.)
    for op in graph.get_op_nodes():

        # Apply extenders to nodes in source graph
        if op.type in Extender.registered_ops:
            Extender.get_extender_class_by_name(op.type).extend(op)
        else:
            log.debug('Extender for node {} with type={} not found, please note.'.format(op.name, op.type))

        # Add node with necessary type and extended attrs in new graph
        op_type = op.soft_get('type_to_create', op.type)

        if op_type in custom_ops:
            node = custom_ops[op_type](new_graph, op.attrs()).create_node()
        else:
            assert op_type in Op.registered_ops, 'Operation {} not found in MO operations, ' \
                                                 'please check it!'.format(op_type)
            node = Op.get_op_class_by_name(op_type)(new_graph, op.attrs()).create_node()

        if op.has_and_set('need_copy_input_blobs'):
            copy_input_blobs(op, node)

        # Collect node connections
        mapping_of_old_idx_into_new[op.id] = node.id
        node_connections[op.id] = collect_node_outputs(op)

    # Restore connections in new graph
    for input_node_idx, its_outputs in list(node_connections.items()):
        for out_port_idx, out_port_dest in its_outputs.items():
            for dest_in_port_idx, dest_node_idx in out_port_dest:
                src = Node(new_graph, mapping_of_old_idx_into_new[input_node_idx])
                dst = Node(new_graph, mapping_of_old_idx_into_new[dest_node_idx])
                src.out_port(out_port_idx).connect(dst.in_port(dest_in_port_idx))

    # Nodes postprocessing stage in new graph
    for op in new_graph.get_op_nodes():
        if op.soft_get('type') in postprocessing_op_nodes:
            postprocessing_op_nodes[op.type](op)

    # clean up graph to shape inference
    new_graph.clean_up()

    return new_graph
