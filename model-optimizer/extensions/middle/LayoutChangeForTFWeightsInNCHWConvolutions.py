# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.transpose import Transpose
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import PermuteAttrs


class LayoutChangeForTFWeightsInNCHWConvolutions(MiddleReplacementPattern):
    """
    Inserts transpose before Convolution (Deconvolution) weights input
    to align with [C_OUT, C_IN, H, W] layout ([C_IN, C_OUT, H, W] for Deconvolution).
    IE weights layout for convolution is [C_OUT, C_IN, H, W]
    while for TF it's always [H, W, C_IN, C_OUT] even if data_format='NCHW'.
    Therefore need to transpose weights input even if
    --disable_nhwc_to_nchw=True and graph['layout']='NCHW'.

    NOTE: when --force_nhwc_to_nchw will be introduced need to update  graph_condition (xxx-51259)
    Limitations: networks with mixed NCHW and NHWC convolutions probably will not be converted properly.
    """
    enabled = True
    force_shape_inference = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NCHW']

    def run_after(self):
        from extensions.middle.pass_separator import PostMiddleStart
        return [PostMiddleStart]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes():
            if node.soft_get('type') in ['Convolution', 'Deconvolution']:
                node_name = node.soft_get('name', node.id)
                assert node.has('get_weights_permute') and isinstance(node['get_weights_permute'], PermuteAttrs.Permutation), \
                    'node {} does not contain weights permutation attribute'.format(node_name)

                transpose_name = node.soft_get('name', node.id) + '/Transpose'
                transpose = create_op_with_const_inputs(graph, Transpose, {1: node['get_weights_permute'].perm},
                                                        {'name': transpose_name, 'override_output_shape': True})
                node.in_port(1).get_connection().insert_node(transpose)
