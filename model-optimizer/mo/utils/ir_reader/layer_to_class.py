# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os

import numpy as np

from extensions.back.TopKNormalizer import TopKNormalizer
from extensions.middle.FakeSplitOutputs import AddFakeOutputsToSplit
from extensions.ops.Cast import Cast
from extensions.ops.ReduceOps import ReduceOp
from extensions.ops.activation_ops import Activation
from extensions.ops.dft import FFTBase
from extensions.ops.elementwise import Elementwise, UnaryElementwise, LogicalElementwise, BiasAdd, Div, Mul, Pow, Sub
from extensions.ops.embedding_bag import EmbeddingBagBase
from extensions.ops.loop import Loop
from extensions.ops.psroipooling import DeformablePSROIPoolingOp
from extensions.ops.scatter import Scatter
from extensions.ops.scatternd import ScatterNDBase
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

# Operations not registered in collect_ops() function
custom_ops = {
    'AvgPool': Pooling,
    'BiasAdd': BiasAdd,
    'Convert': Cast,
    'ConvolutionBackpropData': Deconvolution,
    'DeformablePSROIPooling': DeformablePSROIPoolingOp,
    'Divide': Div,
    'GroupConvolution': Convolution,
    'GroupConvolutionBackpropData': Deconvolution,
    'Loop': Loop,
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
    update_registration(classes=[Op, Activation, Elementwise, UnaryElementwise, LogicalElementwise,
                                 EmbeddingBagBase, ReduceOp, Scatter, ScatterNDBase, FFTBase],
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
            node = Node(graph, u)
            num_of_in_nodes = len(node.in_nodes())
            decremented_number = d['out'] - num_of_in_nodes
            # Initially Const operation in IR has output port with number 1. But later the behaviour was changed
            # so the output port become 0. This change was made to be consistent with the IR serializer in the IE which
            # generates Const with output port 0. For the backward compatibility reason we need to decrement the Const
            # output port number but for current version this number shouldn't be changed during reading the IR.
            if node.type == 'Const' and d['out'] == 0:
                decremented_number = d['out']
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
    assert 0 in op.in_nodes(), 'Can\'t propagate restored value to Const operation with name: {}, check input ports' \
                               ''.format(op.soft_get('name'))
    assert 0 in op.out_nodes(), 'Can\'t propagate restored value to Const operation with name: {}, check output ports' \
                                ''.format(op.soft_get('name'))

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
    if op['element_type'] in ['u4', 'i4']:
        # Packed data types are custom from numpy perspective.
        # Shape from the IR is incompatible with numpy value we store.
        op['value'] = value
        op['force_type'] = op['element_type'].upper()
        op['force_shape'] = op.shape.copy()
    else:
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
    # The only way GroupConvolution with 'group' = 1 appears in IR is by converting from TF DepthwiseConv2dNative.
    # In this case we need to specify 'op' parameter for the
    # extensions.back.ConvolutionNormalizer.ConvolutionWithGroupsResolver to work properly.
    # Otherwise  there will be 'Convolution' instead 'GroupConvolution' in restored IR, since 'GroupConvolution' is
    # extended as node with 'type' = 'Convolution' by IR reader
    if group == 1:
        op['op'] = 'DepthwiseConv2dNative'
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
    'TopK': TopKNormalizer.normalize_outputs,
    # Call normalize Split outputs for generated IR by ir-reader
    'Split': AddFakeOutputsToSplit.split_normalize_outputs,
    'VariadicSplit': AddFakeOutputsToSplit.split_normalize_outputs,
}


def restore_tensor_names(op: Node):
    for out_port in op.ports:
        # op.ports is our internal attribute, dictionary, where keys are numbers of output ports
        # and values are tuples with shape and tensor name:
        # {out_port_idx_1: (out_port_idx_1_shape, out_port_idx_1_tensor_name),
        #  out_port_idx_2: (out_port_idx_2_shape, out_port_idx_2_tensor_name)}
        out_tensor_names = op.ports[out_port][1]

        # handle Constant operations with old style output port numbering
        if op.soft_get('type') == 'Const':
            assert len(op.ports) == 1, 'Something wrong with Constant node: {}, wrong number ' \
                                       'of output ports: {}!'.format(op.soft_get('name'), len(op.ports))
            out_port = 0

        out_port = out_port - len(op.in_nodes())

        if out_tensor_names is not None:
            # handle tensor names with commas and add them to dictionary as separate items
            if out_tensor_names.find(',') >= 0:
                str_to_replace = '<comma_in_tensor_name>'
                out_tensor_names = (out_tensor_names.replace('\\,', str_to_replace)).split(',')
                op.out_node(out_port)['fw_tensor_debug_info'] = []
                for out_tensor_name in out_tensor_names:
                    out_tensor_name = out_tensor_name.replace(str_to_replace, ',')
                    op.out_node(out_port)['fw_tensor_debug_info'].append((out_tensor_name, out_tensor_name))
            else:
                op.out_node(out_port)['fw_tensor_debug_info'] = [(out_tensor_names, out_tensor_names)]


def copy_graph_with_ops(graph: Graph) -> Graph:
    """
    Function to copy graph and apply extenders to appropriate nodes
    :param graph: Graph to copy
    :return:Copied graph with applied extenders
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
        restore_tensor_names(op)

        # operations postprocessing with some special types
        if op.soft_get('type') in postprocessing_op_nodes:
            postprocessing_op_nodes[op.type](op)

    # clean up graph to shape inference
    new_graph.clean_up()

    return new_graph
