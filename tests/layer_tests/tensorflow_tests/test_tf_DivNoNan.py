# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestDivNoNan(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        assert 'y:0' in inputs_info
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape).astype(self.input_type)
        # generate y in way to have zeros
        inputs_data['y:0'] = np.random.randint(-10, 10, y_shape).astype(self.input_type) * \
                           np.random.randint(0, 2, y_shape).astype(self.input_type)
        return inputs_data

    def create_div_no_nan_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            tf.raw_ops.DivNoNan(x=x, y=y)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[10, 20], input_type=np.float32),
        dict(input_shape=[2, 3, 4], input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_div_no_nan_basic(self, params, ie_device, precision, ir_version, temp_dir,
                              use_legacy_frontend):
        self._test(*self.create_div_no_nan_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
