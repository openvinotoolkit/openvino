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

import copy
import logging as log

import networkx as nx
import numpy as np

from mo.front.common.layout import indices_mapping
from mo.front.common.partial_infer.utils import assign_dims_to_weights
from mo.front.extractor import add_attrs_props
from mo.front.extractor import update_ie_fields
from mo.graph.graph import Node, unique_id
from mo.middle.passes.fusing.helpers import get_value_id, get_tensor_id
from mo.middle.passes.shape import repack_fully_connected_weights_nhwc_to_nchw
from mo.middle.pattern_match import apply_pattern
from mo.ops.op import Op, PermuteAttrs
from mo.ops.permute import Permute
from mo.utils.error import Error


def pad_op_transform(graph: nx.MultiDiGraph, match: dict):
    op = match['op']
    pad_op = match['pad_op']
    input_data = pad_op.in_node(0)
    pads = pad_op.in_node(1).value if len(pad_op.in_nodes()) == 2 else pad_op.pads
    op.pad += pads
    op.pad_spatial_shape = op.pad[op.spatial_dims]
    op['auto_pad'] = None
    assert (graph[match['pad_output'].node][match['op'].node][0]['in'] == 0)
    edge_attrs = graph.get_edge_data(match['pad_output'].id, match['op'].id)[0]
    graph.remove_edge(match['pad_output'].id, match['op'].id)
    graph.add_edge(input_data.id, match['op'].id, **{'in': 0, **edge_attrs})


def fuse_pad(graph: nx.MultiDiGraph):
    for op_type in ['Convolution', 'Pooling', 'Deconvolution']:
        apply_pattern(
            graph,
            nodes=[
                ('pad_op', dict(kind='op', op='Pad')),
                ('pad_output', dict(kind='data')),
                ('op', dict(kind='op', type=op_type))],
            edges=[('pad_op', 'pad_output'),
                   ('pad_output', 'op', {'in': 0})],
            action=pad_op_transform
        )


def convert_matmul_to_fully_connected(graph: nx.MultiDiGraph):
    apply_pattern(
        graph,
        nodes=[
            ('matmul', dict(kind='op', op='MatMul')),
            ('output', dict(kind='data'))],
        edges=[('matmul', 'output')],
        action=matmul_to_fully_connected_action
    )


def matmul_to_fully_connected_action(graph: nx.MultiDiGraph, match: dict):
    log.debug('fully_connected_matched')
    matmul = match['matmul']
    input = matmul.in_node(0)
    weights = matmul.in_node(1)
    weights_consumers = graph.out_edges(weights.node)
    log.debug("len(input.shape) = {}, len(weights.shape) = {}, len(weights_consumers) = {}".format(
        len(input.shape) if input.shape is not None else None, len(weights.shape) if not None else None,
        len(weights_consumers) if weights_consumers is not None else None))

    if not (weights.value is not None and
            input.shape is not None and
            len(input.shape) >= 2 and
            weights.shape is not None and
            len(weights.shape) == 2 and
            len(weights_consumers) >= 1):
        matmul['can_be_fused'] = False
        return

    matmul['out-size'] = matmul.out_node().shape[-1]
    matmul['type'] = 'FullyConnected'
    matmul['channel_dims'] = len(match['output'].shape) - 1
    matmul['bias_addable'] = True
    matmul['input_channel_dim'] = 0  # MatMul weights in IO
    matmul['output_channel_dim'] = 1
    matmul['layout'] = 'NHWC'

    assign_dims_to_weights(matmul.in_node(1), None, matmul.input_channel_dim, matmul.output_channel_dim, 2)
    # Do not transpose weights in this pass, it will be done as a separate pass


def transpose_fully_connected_weights(graph: nx.MultiDiGraph):
    transposed_for_IE = 'transposed_for_IE'
    for node in graph.nodes():
        node = Node(graph, node)
        if node.has_valid('type') and node.type == 'FullyConnected':
            weights = node.in_node(1)
            node.in_edge(1)['bin'] = 'weights'
            if weights.has_and_set(transposed_for_IE):
                continue
            # IE accepts weights for fully-connected in OI, but MatMul is in IO, transpose
            weights.value = np.transpose(weights.value)
            weights[transposed_for_IE] = True
            log.debug("Transposed weights {} for FC node {}; weights.shape = {}"
                      "".format(weights.name, node.name, weights.shape))
            weights.shape = np.array(weights.value.shape)


def gemm_to_fully_connected_action(graph: nx.MultiDiGraph, match: dict):
    log.debug('gemm_to_fully_connected_action is triggered')
    gemm = match['gemm']
    A = gemm.in_node(0)
    B = gemm.in_node(1)
    B_consumers = graph.out_edges(B.node)
    C = gemm.in_node(2)
    C_consumers = graph.out_edges(C.node)

    if not (B.value is not None and
            C.value is not None and
            A.shape is not None and
            C.shape.size == 1 and
            not gemm.transpose_a and
            (len(B_consumers) == 1 or not gemm.transpose_b)):
        log.warning('Cannot convert Gemm to FullyConnected')
        return

    if gemm.transpose_b:
        # B.value = B.value.transpose()
        # B.shape = np.array(B.value.shape, dtype=np.int64)
        gemm.transpose_b = 0
    else:
        B.value = B.value.transpose()
        B.shape = np.array(B.value.shape, dtype=np.int64)

    gemm['out-size'] = gemm.out_node().shape[-1]
    gemm['type'] = 'FullyConnected'
    gemm['channel_dims'] = len(match['output'].shape) - 1
    gemm['bias_addable'] = True
    gemm['input_channel_dim'] = 1  # MatMul weights in IO
    gemm['output_channel_dim'] = 0
    gemm['layout'] = 'NCHW'
    gemm.in_edge(1)['bin'] = 'weights'
    gemm.in_edge(2)['bin'] = 'biases'

    assign_dims_to_weights(gemm.in_node(1), None, 1, 0, 2)
    # Do not transpose weights in this pass, it will be done as a separate pass


def convert_gemm_to_fully_connected(graph: nx.MultiDiGraph):
    apply_pattern(
        graph,
        nodes=[
            ('gemm', dict(kind='op', op='Gemm')),
            ('output', dict(kind='data'))],
        edges=[('gemm', 'output')],
        action=gemm_to_fully_connected_action
    )


def muladd_to_scaleshift_action(graph: nx.MultiDiGraph, match: dict):
    mul = match['mul']
    add = match['add']
    output = match['output']

    if mul.soft_get('can_be_scaleshift') is False or add.soft_get('can_be_scaleshift') is False:
        return

    mul_weights_id = get_value_id(mul)
    mul_input_id = get_tensor_id(mul)
    add_weights_id = get_value_id(add)

    if mul_weights_id is None:
        log.debug("Mul->Add to ScaleShift: Mul {} has no weights".format(mul.name))
        return
    if mul_input_id is None:
        log.debug("Mul->Add to ScaleShift: Mul {} has no input".format(mul.name))
        return
    if add_weights_id is None:
        log.debug("Mul->Add to ScaleShift: Add {} has no weights".format(add.name))
        return

    input = mul.in_node(mul_input_id)
    weights = mul.in_node(mul_weights_id)
    bias = add.in_node(add_weights_id)

    # Transform values
    weights.value = np.squeeze(weights.value)
    weights.shape = weights.value.shape

    bias.value = np.squeeze(bias.value)
    bias.shape = bias.value.shape

    # Broadcast weights if they are scalar
    if weights.value.ndim == 0 and bias.value.ndim == 1:
        weights.value = np.full(bias.shape, weights.value.item())
        weights.shape = weights.value.shape

    if bias.shape != weights.shape:
        log.warning('Mul->Add to ScaleShift conversion stoped {} != {}'.format(weights.shape, bias.shape))
        return

    if bias.value.ndim != weights.value.ndim or bias.value.size != weights.value.size:
        log.debug("Skipping Mul->Add to scaleshift or power conversion for nodes {}, {} because of different weights "
                  "and biases".format(mul.name, add.name))
        return

    op_name = "ScaleShift"
    if bias.value.size == 1 and weights.value.size == 1:
        op_name = "Power"

    log.debug("Fusing Mul->Add to {}. Input nodes: {} and {}, bias.shape = {}, weights.shape = {}"
              "".format(op_name, mul.id, add.id, bias.shape, weights.shape))

    graph.remove_edge(input.node, mul.id)
    graph.remove_edge(weights.node, mul.id)
    graph.remove_edge(bias.node, add.id)
    graph.remove_edge(add.node, output.id)

    op_node = unique_id(graph, mul.name + '/Fused{}_'.format(op_name))
    if op_name == 'ScaleShift':
        graph.add_node(op_node, **add_attrs_props(dict(kind='op', precision="FP32", type=op_name, name=op_node,
                                                       op=op_name, data_type=input.data_type)))
        update_ie_fields(graph.node[op_node])
        graph.add_edges_from([
            (input.node, op_node, {'in': 0}),
            (weights.node, op_node, {'in': 1, 'bin': 'weights'}),
            (bias.node, op_node, {'in': 2, 'bin': 'biases'}),
            (op_node, output.node, {'out': 0})
        ])
    else:
        graph.add_node(op_node, **add_attrs_props(dict(kind='op', precision="FP32", type=op_name, name=op_node,
                                                       op=op_name, data_type=input.data_type, power=1,
                                                       scale=weights.value.item(), shift=bias.value.item())))
        update_ie_fields(graph.node[op_node])
        graph.add_edges_from([
            (input.node, op_node, {'in': 0}),
            (op_node, output.node, {'out': 0})
        ])

    return


def convert_muladd_to_scaleshift_or_power(graph: nx.MultiDiGraph):
    apply_pattern(
        graph,
        nodes=[
            ('input', dict(kind='data')),
            ('weights', dict(kind='data')),
            ('bias', dict(kind='data')),
            ('mout', dict(kind='data')),
            ('output', dict(kind='data')),
            ('mul', dict(kind='op', op='Mul')),
            ('add', dict(kind='op', op='Add'))
        ],
        edges=[
            ('weights', 'mul', {'in': 0}),
            ('input', 'mul', {'in': 1}),
            ('mul', 'mout'),
            ('mout', 'add', {'in': 0}),
            ('bias', 'add', {'in': 1}),
            ('add', 'output'),
        ],
        action=muladd_to_scaleshift_action
    )


def batch_norm_fuse_action(graph: nx.MultiDiGraph, match: dict):
    """
    Multiply convolution kernel by batch normalization coefficient and remove mul op.
    """
    if match['norm'].value is None or match['kernel'].value is None:
        # cannot fuse non-const normalization coefficients
        return
    if len(graph.out_edges(match['conv_output'].node)) > 1 or len(graph.out_edges(match['kernel'].node)) > 1:
        # we cannot modify original kernel or convolution, if they are used multiple times
        # TODO make a copy of conv and kernel instead of this check
        return
    match['kernel'].value = match['kernel'].value * match['norm'].value
    graph.remove_edge(match['conv_output'].node, match['mul'].node)
    graph.remove_edge(match['mul'].node, match['mul_output'].node)
    # graph.remove_node(match['mul'].node)    # if we remove a node, next iteration over isomorphisms gives an error
    graph.add_edge(match['conv'].node, match['mul_output'].node, out=0)


def batch_norm_fuse(graph: nx.MultiDiGraph):
    apply_pattern(
        graph,
        nodes=[
            ('kernel', dict(kind='data')),
            ('conv', dict(kind='op', op='Conv2D')),
            ('conv_output', dict(kind='data')),
            ('norm', dict(kind='data')),
            ('mul', dict(kind='op', op='Mul')),
            ('mul_output', dict(kind='data'))],
        edges=[
            ('kernel', 'conv', {'in': 1}),
            ('conv', 'conv_output'),
            ('conv_output', 'mul', {'in': 0}),  # TODO get rid of explicit input port number, mul is a commutative op
            ('norm', 'mul', {'in': 1}),  # TODO get rig of explicit input port number, mul is a commutative op
            ('mul', 'mul_output')],
        action=batch_norm_fuse_action
    )
    return graph


def convert_add_to_scaleshift(graph: nx.MultiDiGraph):
    for n in list(graph.nodes()):
        node = Node(graph, n)
        if node.has('op') and (node.op == 'BiasAdd' or node.op == 'Add') and len(node.in_nodes()) == 2:
            tensor_id, value_id = get_tensor_id(node), get_value_id(node)
            if tensor_id is not None and value_id is not None and node.soft_get('can_be_scaleshift') is not False:
                node['type'] = 'ScaleShift'
                node['op'] = 'ScaleShift'
                node.in_node(value_id).value = np.squeeze(node.in_node(value_id).value)
                node.in_node(value_id).shape = node.in_node(value_id).value.shape

                # if the node was created with eltwise then it has attribute 'operation' which should be removed from the IR
                if node.has('operation'):
                    del graph.node[n]['operation']

                bias_data = node.in_node(value_id)
                graph[bias_data.node][node.node][0]['in'] = 2
                graph[bias_data.node][node.node][0]['bin'] = 'biases'

                input_data = node.in_node(tensor_id)
                graph[input_data.node][node.node][0]['in'] = 0

                update_ie_fields(graph.node[node.id])

                weights_id = unique_id(graph, 'weights_')
                graph.add_node(weights_id, **add_attrs_props(
                    dict(kind='data', precision="FP32", name=weights_id, value=None, shape=None, data_type=None,
                         infer=None)))
                wnode = Node(graph, weights_id)

                wnode['value'] = np.full_like(bias_data.value, 1, dtype=np.float32)
                wnode['shape'] = np.array(wnode['value'].shape)

                graph.add_edges_from([
                    (weights_id, node.node, {'in': 1, 'bin': 'weights'}),
                ])


def convert_mul_to_scaleshift(graph: nx.MultiDiGraph):
    for n in list(graph.nodes()):
        node = Node(graph, n)
        if node.has('op') and node.op == 'Mul' and len(node.in_nodes()) == 2:
            tensor_id, value_id = get_tensor_id(node), get_value_id(node)
            if tensor_id is not None and value_id is not None and node.soft_get('can_be_scaleshift') is not False:
                node['type'] = 'ScaleShift'
                node['op'] = 'ScaleShift'
                node.in_node(value_id).value = np.squeeze(node.in_node(value_id).value)
                node.in_node(value_id).shape = node.in_node(value_id).value.shape

                # if the node was created with eltwise then it has attribute 'operation' which should be removed from the IR
                if node.has('operation'):
                    del graph.node[n]['operation']

                scale_data = node.in_node(value_id)
                graph[scale_data.node][node.node][0]['in'] = 1
                graph[scale_data.node][node.node][0]['bin'] = 'weights'

                input_data = node.in_node(tensor_id)
                graph[input_data.node][node.node][0]['in'] = 0

                update_ie_fields(graph.node[node.id])

                bias_id = unique_id(graph, 'bias_')
                graph.add_node(bias_id, **add_attrs_props(
                    dict(kind='data', precision="FP32", name=bias_id, value=None, shape=None, data_type=None,
                         infer=None)))
                wnode = Node(graph, bias_id)

                wnode['value'] = np.full_like(scale_data.value, 0, dtype=np.float32)
                wnode['shape'] = np.array(wnode['value'].shape)

                graph.add_edges_from([
                    (bias_id, node.node, {'in': 2, 'bin': 'biases'}),
                ])


def convert_nasnet_action(graph: nx.MultiDiGraph, matches: dict):
    """
    This function converts speciefic for NasNet topology subgraph Pad->StridedSlice->AvgPool to Conv->Crop->AvgPool
    """
    input = matches['input']
    output = matches['output']

    pad_op = matches['pad_op']
    pad_const = matches['pad_const']
    pad_out = matches['pad_out']

    sslice = matches['sslice']
    sslice_out = matches['sslice_out']
    begin = []
    end = []
    stride = []
    for s in sslice.slices:
        begin.append(s.start)
        end.append(s.stop)
        stride.append(s.step)

    avg_pool = matches['avg_pool']

    if not np.array_equal(pad_const.value, np.array([[0, 0], [0, 1], [0, 1], [0, 0]])):
        log.error(" Pad values doesn't match!")
        return

    if not np.array_equal(begin, np.array([0, 1, 1, 0])):
        log.error("StridedSlice has wrong begin")
        return

    if sslice.end_mask != 15 or sslice.begin_mask != 9:
        log.error("StridedSlice has wrong masks")
        return

    # Cut Smth-x->Pad->StrudedSlice-x->AvgPool
    graph.remove_edge(input.id, pad_op.id)
    graph.remove_edge(sslice.id, sslice_out.id)

    # Pad -> Conv
    conv_node = unique_id(graph, pad_op.name + '/Conv_')
    conv_weights_node = unique_id(graph, pad_op.name + '/ConvW_')
    conv_weights = np.ones((1, 1, input.shape[3], 1))
    conv_output = unique_id(graph, pad_op.name + '/ConvOut_')
    output_shape = np.array([input.shape[0], input.shape[1] + 1, input.shape[2] + 1, input.shape[3]])

    graph.add_node(conv_node,
                   **add_attrs_props(dict(kind='op', precision="FP32", type='Convolution', name=conv_node, op='Conv2D',
                                          stride=np.array([1, 1, 1, 1]), dilation=np.array([1, 1, 1, 1]),
                                          group=input.shape[3], bias_addable=True, bias_term=False,
                                          spatial_dims=np.array([1, 2]),
                                          kernel_spatial=np.array([1, 1]),
                                          pad=np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), output_shape=output_shape,
                                          channel_dims=np.array([3]))))

    graph.add_node(conv_weights_node, **add_attrs_props(
        dict(kind='data', precision="FP32", name=conv_weights_node, value=np.array(conv_weights),
             shape=np.array(conv_weights.shape),
             data_type=input.data_type, infer=None,
             spatial_dims=np.array([0, 1]),
             input_channel_dim=np.array(2),
             output_channel_dim=np.array(3),
             dims_number=np.array(4), can_be_bias=True)))
    graph.add_node(conv_output, **add_attrs_props(
        dict(kind='data', precision="FP32", name=conv_output, value=None, shape=output_shape,
             data_type=input.data_type)))

    # StridedSlice -> Crop
    Crop = Op.get_op_class_by_name('Crop')
    crop = Crop(graph, dict(name=sslice.name + '/Crop_', axis=np.array([1, 2]),
                            dim=np.array([output_shape[1] - 1, output_shape[2] - 1]), offset=np.array([1, 1])))
    crop.create_node_with_data([Node(graph, conv_output)], data_nodes=sslice_out)
    # graph.add_node(crop_node, **add_attrs_props(dict(kind='op', precision="FP32", type='Crop', name=crop_node,
    #                                                 op='Crop', axis=[1,2], dim=[output_shape[1]-1, output_shape[2]-1], offset=[1,1])))

    # Connect : Conv->Crop->AvgPool
    graph.add_edges_from([
        (input.id, conv_node, {'in': 0}),
        (conv_weights_node, conv_node, {'in': 1, 'bin': 'weights'}),
        (conv_node, conv_output, {'out': 0}),
    ])
    update_ie_fields(graph.node[conv_node], graph.graph['ir_version'])


def convert_nasnet(graph: nx.MultiDiGraph):
    apply_pattern(
        graph,
        nodes=[
            ('input', dict(kind='data')),
            ('pad_const', dict(kind='data')),
            ('pad_op', dict(kind='op', op='Pad')),
            ('pad_out', dict(kind='data')),

            ('begin', dict(kind='data')),
            ('end', dict(kind='data')),
            ('stride', dict(kind='data')),

            ('sslice', dict(kind='op', op='StridedSlice')),
            ('sslice_out', dict(kind='data')),

            ('avg_pool', dict(kind='op', op='AvgPool')),
            ('output', dict(kind='data')),
        ],
        edges=[
            ('input', 'pad_op', {'in': 0}),
            ('pad_const', 'pad_op', {'in': 1}),
            ('pad_op', 'pad_out'),

            ('begin', 'sslice', {'in': 1}),
            ('end', 'sslice', {'in': 2}),
            ('stride', 'sslice', {'in': 3}),

            ('pad_out', 'sslice', {'in': 0}),
            ('sslice', 'sslice_out'),

            ('sslice_out', 'avg_pool', {'in': 0}),
            ('avg_pool', 'output')
        ],
        action=convert_nasnet_action
    )
    return graph


def dilated_convolution_action(graph: nx.MultiDiGraph, match: dict):
    conv = match['conv']
    stb = match['space_to_batch']
    bts = match['batch_to_space']

    block_size = match['stb_bs']

    input = match['input']
    output = match['output']
    stb_out = match['stb_output']
    conv_out = match['conv_output']

    in_edge_attrs = graph.get_edge_data(input.id, stb.id)[0]
    out_edge_attrs = graph.get_edge_data(bts.id, output.id)[0]

    graph.remove_edge(input.id, stb.id)
    graph.remove_edge(stb_out.id, conv.id)
    graph.remove_edge(conv.id, conv_out.id)
    graph.remove_edge(bts.id, output.id)

    conv.dilation[conv.spatial_dims] = block_size.value

    pad = match['stb_pad'].value - match['bts_crop'].value
    conv.pad[conv.spatial_dims] = [[pad[x][0], pad[x][1]] for x in range(len(pad))]
    conv['auto_pad'] = None

    graph.add_edges_from([
        (input.id, conv.id, {'in': 0, **in_edge_attrs}),
        (conv.id, output.id, {'out': 0, **out_edge_attrs}),
    ])


def convert_dilated_convolution(graph: nx.MultiDiGraph):
    for op in ['Conv2D', 'DepthwiseConv2dNative', 'Conv3D']:
        apply_pattern(
            graph,
            nodes=[
                ('conv', dict(kind='op', op=op)),
                ('space_to_batch', dict(kind='op', op='SpaceToBatchND')),
                ('batch_to_space', dict(kind='op', op='BatchToSpaceND')),
                ('input', dict(kind='data')),
                ('output', dict(kind='data')),
                ('conv_output', dict(kind='data')),
                ('stb_output', dict(kind='data')),
                ('stb_bs', dict(kind='data')),
                ('stb_pad', dict(kind='data')),
                ('bts_bs', dict(kind='data')),
                ('bts_crop', dict(kind='data'))
            ],
            edges=[
                ('input', 'space_to_batch', {'in': 0}),
                ('stb_bs', 'space_to_batch', {'in': 1}),
                ('stb_pad', 'space_to_batch', {'in': 2}),
                ('space_to_batch', 'stb_output', {'out': 0}),
                ('stb_output', 'conv', {'in': 0}),
                ('conv', 'conv_output', {'out': 0}),
                ('conv_output', 'batch_to_space', {'in': 0}),
                ('bts_bs', 'batch_to_space', {'in': 1}),
                ('bts_crop', 'batch_to_space', {'in': 2}),
                ('batch_to_space', 'output', {'out': 0}),
            ],
            action=dilated_convolution_action
        )


def convert_multi_input_conv(graph: nx.MultiDiGraph):
    for node in list(graph.nodes()):
        node = Node(graph, node)
        if node.kind == 'op' and node.op == 'ConvND':
            node.op = 'Conv2D'
            if node.bias_term == True:
                num_inputs = len(node.in_nodes()) - 2
                w_node = node.in_node(len(node.in_nodes()) - 2)
                b_node = node.in_node(len(node.in_nodes()) - 1)
            else:
                num_inputs = len(node.in_nodes()) - 1
                w_node = node.in_node(len(node.in_nodes()) - 1)

            for i in range(0, num_inputs - 1):
                in_i = node.in_node(i)
                out_i = node.out_node(i)
                conv_id = unique_id(graph, node.id + '__')
                graph.add_node(conv_id, **copy.deepcopy(node.get_attrs()))
                new_conv = Node(graph, conv_id)
                new_conv.name = conv_id

                graph.remove_edge(in_i.id, node.id)
                graph.remove_edge(node.id, out_i.id)
                graph.add_edges_from([
                    (w_node.id, conv_id, {'in': 1, 'bin': 'weights'}),
                ])

                if node.bias_term == True:
                    graph.add_edges_from([
                        (b_node.id, conv_id, {'in': 2, 'bin': 'biases'}),
                    ])

                graph.add_edges_from([
                    (in_i.id, conv_id, {'in': 0}),
                ])
                graph.add_edge(conv_id, out_i.id, out=3)
