# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(23345)


class TestLogicalBinaryOps(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']

        inputs_data = {}
        inputs_data['x:0'] = rng.choice([True, False], x_shape).astype(bool)
        inputs_data['y:0'] = rng.choice([True, False], y_shape).astype(bool)
        return inputs_data

    def create_logical_binary_ops_net(self, x_shape, y_shape, op_type):
        op_type_map = {
            'LogicalAnd': tf.raw_ops.LogicalAnd,
            'LogicalOr': tf.raw_ops.LogicalOr,
        }

        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(bool, x_shape, 'x')
            y = tf.compat.v1.placeholder(bool, y_shape, 'y')
            op_type_map[op_type](x=x, y=y, name=op_type)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('x_shape', [[], [4], [3, 4], [2, 3, 4]])
    @pytest.mark.parametrize('y_shape', [[2, 3, 4]])
    @pytest.mark.parametrize("op_type", ['LogicalAnd', 'LogicalOr'])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_logical_binary_op(self, x_shape, y_shape, op_type,
                               ie_device, precision, ir_version,
                               temp_dir, use_legacy_frontend):
        self._test(*self.create_logical_binary_ops_net(x_shape=x_shape, y_shape=y_shape, op_type=op_type),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)
