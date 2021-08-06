# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.graph.graph import Node
from extensions.ops.transpose import Transpose
from mo.ops.convolution import Convolution
from mo.ops.deconvolution import Deconvolution
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.utils.error import Error
from mo.front.tf.graph_utils import create_op_with_const_inputs


class PermuteTFNCHWConvWeights(BackReplacementPattern):
    """
    Inserts transpose before Convolution weights input to align with [C_OUT, C_IN, H, W] layout.
    IE weights layout is [C_OUT, C_IN, H, W] while for TF
    it's always [H, W, C_IN, C_OUT] even if Conv data_format == 'NCHW'.
    Need to transpose weights input even if graph['layout'] == 'NCHW'.

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
            if node.op in ['Conv2D', 'Conv2DBackpropInput']:
                permutation = [3, 2, 0, 1]  # todo
                transpose_name = node.soft_get('name', node.id) + '/Transpose'

                transpose = create_op_with_const_inputs(graph, Transpose, {1: permutation},
                                                        {'name': transpose_name, 'override_output_shape': True})
                node.in_port(1).get_connection().insert_node(transpose)
