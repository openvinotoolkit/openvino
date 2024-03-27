# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape


class FusedBatchNormTraining(MiddleReplacementPattern):
    """
    Transformation looks for the BatchNorm layers in training mode and does the following:
    1. Fuses batch dimension with one of the spatial dimensions of the input to BatchNorm because batch normalization is
    performed over batch dimension also (per channel(features) dimension).
    2. Inserts MVN layer.
    3. Reshape MVN output back to the original one.
    """
    enabled = True
    replacement_id = "Fused_Batch_Norm_is_training_true"
    force_shape_inference = True
    force_clean_up = True
    # transformation works for the NHWC layout because transformation inserts Reshape to fuse N and H dimensions
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(kind='op', op=lambda op: op in ['FusedBatchNorm', 'FusedBatchNormV2', 'FusedBatchNormV3'],
                            is_training=True))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['op']
        node_name = node.soft_get('name', node.id)
        node.is_training = False

        shape = node.in_port(1).data.get_shape()
        assert shape is not None, 'The shape of scale input of the BatchNorm node {} is not defined'.format(node.name)

        bn_mean = Const(graph, {'name': node_name + '/mean', 'value': np.zeros(shape, dtype=np.float32),
                                'override_output_shape': True}).create_node()
        bn_std = Const(graph, {'name': node_name + '/std', 'value': np.ones(shape, dtype=np.float32),
                               'override_output_shape': True}).create_node()
        node.in_port(3).get_connection().set_source(bn_mean.out_port(0))
        node.in_port(4).get_connection().set_source(bn_std.out_port(0))

        # save the original shape
        original_shape = Shape(graph, {'name': node.in_port(0).get_source().node.soft_get('name')}).create_node()
        original_shape.in_port(0).connect(node.in_port(0).get_source())

        input_rank = len(node.in_port(0).data.get_shape())
        rng = create_op_with_const_inputs(graph, Range,
                                          {0: int64_array(1), 1: int64_array(input_rank - 1), 2: int64_array(1)},
                                          {'name': node_name + '/Range', 'output_type': np.int64})
        mvn = MVN(graph, {'name': node_name + '/mvn_', 'eps': node.soft_get('eps', 1e-6), 'eps_mode': 'inside_sqrt',
                          'normalize_variance': 1, 'override_output_shape': True}).create_node()
        node.in_port(0).get_connection().insert_node(mvn)
        mvn.in_port(1).connect(rng.out_port(0))

        reshape_4d = create_op_node_with_second_input(graph, Reshape, int64_array([1, -1, 0, 0]),
                                                      {'override_output_shape': True,
                                                       'name': node_name + '/fused_batch_and_channels'})
        mvn.in_port(0).get_connection().insert_node(reshape_4d)

        # restore original shape
        reshape_back = Reshape(graph, {'name': node_name + '/restore_shape',
                                       'override_output_shape': True}).create_node()
        reshape_back.in_port(1).connect(original_shape.out_port(0))
        mvn.out_port(0).get_connection().insert_node(reshape_back)
