# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSelectV2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'cond:0' in inputs_info, "Test error: inputs_info must contain `cond`"
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
        cond_shape = inputs_info['cond:0']
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['cond:0'] = np.random.randint(0, 2, cond_shape).astype(bool)
        inputs_data['x:0'] = np.random.randint(-100, 100, x_shape).astype(np.float32)
        inputs_data['y:0'] = np.random.randint(-100, 100, y_shape).astype(np.float32)
        return inputs_data

    def create_select_v2_net(self, cond_shape, x_shape, y_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            cond = tf.compat.v1.placeholder(tf.bool, cond_shape, 'cond')
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            y = tf.compat.v1.placeholder(tf.float32, y_shape, 'y')
            tf.raw_ops.SelectV2(condition=cond, t=x, e=y, name='select_v2')
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(cond_shape=[3, 1], x_shape=[3, 1], y_shape=[3, 1]),
        dict(cond_shape=[], x_shape=[2], y_shape=[3, 2]),
        dict(cond_shape=[4], x_shape=[3, 2, 1], y_shape=[2, 4]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_select_v2_basic(self, params, ie_device, precision, ir_version, temp_dir,
                             use_legacy_frontend):
        if use_legacy_frontend:
            pytest.skip("Select tests are not passing for the legacy frontend.")
        self._test(*self.create_select_v2_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
