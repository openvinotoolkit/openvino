# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestAngle(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'y:0' in inputs_info
        assert 'x:0' in inputs_info
        y_shape = inputs_info['y:0']
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['y:0'] = np.random.rand(*y_shape).astype(self.input_type) - np.random.rand(*y_shape).astype(self.input_type)
        inputs_data['x:0'] = np.random.rand(*x_shape).astype(self.input_type) - np.random.rand(*x_shape).astype(self.input_type)
        return inputs_data

    def create_angle_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            complex = tf.raw_ops.Complex(real=x, imag=y)
            tf.raw_ops.Angle(input=complex)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1, 2], input_type=np.float32),
        dict(input_shape=[2, 3, 4], input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_angle(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(*self.create_angle_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
