# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestApproximateEqual(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info
        assert 'y' in inputs_info
        x_shape = inputs_info['x']
        y_shape = inputs_info['y']
        inputs_data = {}
        inputs_data['x'] = np.random.uniform(0, 10, x_shape).astype(self.input_type)
        inputs_data['y'] = np.random.uniform(0, 10, y_shape).astype(self.input_type)
        inputs_data['tolerance'] = np.random.uniform(0, 1, ()).astype(self.input_type)
        return inputs_data

    def create_approximate_equal_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            tolerance = tf.compat.v1.placeholder(input_type, (), 'tolerance')
            tf.raw_ops.ApproximateEqual(x=x, y=y, tolerance=tolerance)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[10, 20], input_type=np.float32),
        dict(input_shape=[2, 3, 4], input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_approximate_equal_basic(self, params, ie_device, precision, ir_version, temp_dir,
                              use_new_frontend):
        self._test(*self.create_approximate_equal_net(**params), ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, **params)
