# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, connect, \
    shaped_data, connect_front

from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc


class TestTFScatterND(CommonTFLayerTest):
    def create_tf_scatternd_placeholder_const_net(self, x_shape, indices, updates, ir_version, use_new_frontend):

        #
        #   Create Tensorflow model
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = x_shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)

            x = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')
            tf_indices = tf.constant(indices)
            tf_updates = tf.constant(updates)

            scatter_nd = tf.scatter_nd(tf_indices, tf_updates, tf.shape(x), name="Operation")
            rs = tf.add(x,scatter_nd)
            tf.nn.relu(rs)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        ref_net = None
        indicies_np = int64_array(indices)
        updates_np = float32_array(updates)
        if check_ir_version(10, None, ir_version) and not use_new_frontend:
            const_for_layer_tests = lambda name, value, shape, shape1: {
                **{name + '_dd': {'kind': 'data', 'value': value, 'shape': shape1}},
                **{name: {'kind': 'op', 'type': 'Const'}},
                **shaped_data(name + '_d', shape)}

            connect_const_for_layer_tests = lambda first_tensor_name, second_tensor_name: [
                *connect_front(first_tensor_name + '_dd', first_tensor_name),
                *connect(first_tensor_name, second_tensor_name)]

            nodes_attributes = {
                **regular_op_with_shaped_data('input', x_shape, {'type': 'Parameter'}),
                **const_for_layer_tests('indices', indicies_np, indicies_np.shape, indicies_np.shape),
                **const_for_layer_tests('updates', updates_np, updates_np.shape, updates_np.shape),
                **const_for_layer_tests('zero_tensor', float32_array(0.0), int64_array([]), int64_array([])),
                **regular_op_with_shaped_data('broadcast', x_shape, {'type': 'Broadcast'}),
                **regular_op_with_shaped_data('shapeof', x_shape, {'type': 'Shapeof'}),
                **regular_op_with_shaped_data('scatter_nd', x_shape, {'type': 'ScatterNDUpdate'}),
                **regular_op_with_shaped_data('relu', x_shape, {'type': 'ReLU'}),
                **regular_op_with_shaped_data('sum', x_shape, {'type': 'Add'}),
                **regular_op_with_shaped_data('result', x_shape, {'type': 'Result'}),

            }

            ref_net = build_graph(nodes_attributes,
                                  [*connect('input', '0:shapeof'),
                                   *connect_const_for_layer_tests('zero_tensor', '0:broadcast'),
                                   *connect('shapeof', '1:broadcast'),
                                   *connect('broadcast', '0:scatter_nd'),
                                   *connect_const_for_layer_tests('indices', '1:scatter_nd'),
                                   *connect_const_for_layer_tests('updates', '2:scatter_nd'),
                                   *connect('input', '0:sum'),
                                   *connect('scatter_nd', '1:sum'),
                                   *connect('sum', '0:relu'),
                                   *connect('relu', '0:result')])

        return tf_net, ref_net

    test_data = [pytest.param(
        dict(x_shape=[8], indices=[[4], [3], [1], [7]], updates=[9.0, 10.0, 11.0, 12.0]),
        marks=pytest.mark.precommit),
        dict(x_shape=[4, 4, 4], indices=[[0], [2]], updates=\
            [[[5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0], [7.0, 7.0, 7.0, 7.0], [8.0, 8.0, 8.0, 8.0]],
             [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]])
    ]

    @pytest.mark.parametrize("params", test_data)
    # @pytest.mark.nightly
    def test_tf_scatter_nd(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend):
        self._test(*self.create_tf_scatternd_placeholder_const_net(**params, ir_version=ir_version,
                                                                   use_new_frontend=use_new_frontend),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
