# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from copy import deepcopy

import numpy as np

from extensions.middle.LayoutChangeForTFNCHWConvolutions import LayoutChangeForTFWeightsInNCHWConvolutions
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.eliminate import shape_inference
from mo.middle.passes.infer import partial_infer
from mo.ops.convolution import Convolution
from mo.ops.deconvolution import Deconvolution
from mo.ops.op import PermuteAttrs
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_empty_data, connect
from unit_tests.utils.graph import valued_const_with_data, shaped_parameter


class LayoutChangeForTFNCHWConvTests(unittest.TestCase):
    def test_convolution(self):
        graph, graph_ref = build_conv_test_graphs([1, 3, 100, 100], [4, 4, 3, 10])
        LayoutChangeForTFWeightsInNCHWConvolutions().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=False)
        self.assertTrue(flag, resp)

    def test_deconvolution(self):
        graph, graph_ref = build_deconv_test_graphs([1, 3, 100, 100], [4, 4, 3, 64], [1, 3, 103, 103])
        LayoutChangeForTFWeightsInNCHWConvolutions().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=False)
        self.assertTrue(flag, resp)


def build_deconv_test_graphs(input_shape, weights_shape, out_shape=(), layout='NCHW'):
    deconv_node = regular_op_with_empty_data('deconv', {
        'type': 'Deconvolution', 'kind': 'op',
        'infer': Deconvolution.infer,
        'layout': layout,
        'auto_pad': 'valid',
        'stride': np.array([1, 1, 1, 1]),
        'spatial_dims': np.array([2, 3]),

        'input_feature_channel': 3,
        'output_feature_channel': 2,
        'channel_dims': np.array([1]),
        'batch_dims': np.array([0]),

        'get_weights_permute': PermuteAttrs.Permutation(perm=int64_array([3, 2, 0, 1]),
                                                        inv=int64_array([2, 3, 1, 0])),

        # placeholders: these values will be overwritten by shape_infer
        'pad_spatial_shape': None,
        'pad': None,
        'output': None,
        'output_shape': None,
        'op': None,
    })

    nodes = {
        **valued_const_with_data('out_shape', int64_array(out_shape)),
        **shaped_parameter('input', int64_array(input_shape)),
        **shaped_parameter('weights', int64_array(weights_shape)),  # shaped_const_with_data will fill with None
        **deconv_node,
        **result('res'),
    }
    edges = [
        *connect('input', '0:deconv'),
        *connect('weights', '1:deconv'),
        *connect('out_shape', '2:deconv'),
        *connect('deconv', 'res'),
    ]

    graph = build_graph(nodes, edges, nodes_with_edges_only=True)
    graph = partial_infer(graph)

    deconv_node_ref = deepcopy(deconv_node)
    # when transformation inserts Transpose real 'input_feature_channel', 'output_feature_channel' are also changed
    deconv_node_ref['deconv'].update({'output_feature_channel': 1,
                                      'input_feature_channel': 0})

    nodes.update({
        **regular_op_with_empty_data('transpose', {'type': 'Transpose',
                                                   'kind': 'op',
                                                   'infer': Transpose.infer}),
        **valued_const_with_data('input_order', int64_array([3, 2, 0, 1])),
        **deconv_node_ref,
    })

    edges_ref = [
        *connect('input', '0:deconv'),
        *connect('weights', '0:transpose'),
        *connect('input_order', '1:transpose'),
        *connect('transpose', '1:deconv'),
        *connect('out_shape', '2:deconv'),
        *connect('deconv', 'res'),
    ]

    graph_ref = build_graph(nodes, edges_ref, nodes_with_edges_only=True)
    graph_ref = partial_infer(graph_ref)
    return graph, graph_ref


def build_conv_test_graphs(input_shape, weights_shape, layout='NCHW'):
    conv_node = regular_op_with_empty_data('conv', {
        'type': 'Convolution', 'kind': 'op',
        'infer': Convolution.infer,
        'layout': layout,
        'auto_pad': 'valid',

        'input_feature_channel': 2,
        'output_feature_channel': 3,
        'channel_dims': np.array([1]),
        'batch_dims': np.array([0]),

        'get_weights_permute': PermuteAttrs.Permutation(perm=int64_array([3, 2, 0, 1]),
                                                        inv=int64_array([2, 3, 1, 0]))
    })

    nodes = {
        **shaped_parameter('input', int64_array(input_shape)),
        **shaped_parameter('weights', int64_array(weights_shape)),  # shaped_const_with_data will fill with None
        **conv_node,
        **result('res'),
    }
    edges = [
        *connect('input', '0:conv'),
        *connect('weights', '1:conv'),
        *connect('conv', 'res'),
    ]

    graph = build_graph(nodes, edges, nodes_with_edges_only=True)
    graph = partial_infer(graph)

    conv_node_ref = deepcopy(conv_node)

    # when transformation inserts Transpose real 'input_feature_channel', 'output_feature_channel' are also changed
    conv_node_ref['conv'].update({'output_feature_channel': 0,
                                  'input_feature_channel': 1})

    nodes.update({
        **regular_op_with_empty_data('transpose', {'type': 'Transpose',
                                                   'kind': 'op',
                                                   'infer': Transpose.infer}),
        **valued_const_with_data('input_order', int64_array([3, 2, 0, 1])),
        **conv_node_ref,
    })

    edges_ref = [
        *connect('input', '0:conv'),
        *connect('weights', '0:transpose'),
        *connect('input_order', '1:transpose'),
        *connect('transpose', '1:conv'),
        *connect('conv', 'res'),
    ]

    graph_ref = build_graph(nodes, edges_ref, nodes_with_edges_only=True)
    graph_ref = partial_infer(graph_ref)
    return graph, graph_ref
