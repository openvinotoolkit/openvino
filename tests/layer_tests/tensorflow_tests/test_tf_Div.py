# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestDiv(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        assert 'y:0' in inputs_info
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        # generate x and y
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape).astype(self.input_type)
        inputs_data['y:0'] = np.random.randint(1, 10, y_shape)*np.random.choice([-1,1], y_shape)
        return inputs_data

    def create_div_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            tf.raw_ops.Div(x=x, y=y)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[10, 20], input_type=np.float32),
        dict(input_shape=[2, 3, 4], input_type=np.float32),
        pytest.param(dict(input_shape=[8, 5], input_type=np.int32),
                              marks=pytest.mark.xfail(reason='Ticket CVS-132377 - Divide inconsistent behavior on different systems')),
        dict(input_shape=[], input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_div_basic(self, params, ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        self._test(*self.create_div_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
