# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestXlogy(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info
        assert 'y' in inputs_info
        x_shape = inputs_info['x']
        y_shape = inputs_info['y']
        inputs_data = {}
        # x = [-3 ,3] y = [1, 2]
        # generate x in way to have zeros
        inputs_data['x'] = (6 * np.random.random(size=x_shape).astype(np.float32) - 3) * \
                            np.random.randint(2, size=x_shape).astype(np.float32)
        inputs_data['y'] = np.random.random(size=y_shape).astype(np.float32) + 1
        return inputs_data

    def create_xlogy_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            tf.raw_ops.Xlogy(x=x, y=y)
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
    def test_xlogy_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_new_frontend):
        self._test(*self.create_xlogy_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
