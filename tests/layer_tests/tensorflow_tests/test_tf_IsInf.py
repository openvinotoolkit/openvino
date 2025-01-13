# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import mix_array_with_value


class TestIsInf(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `data`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        data = np.random.randint(-50, 50, x_shape).astype(np.float32)
        inputs_data['x:0'] = mix_array_with_value(data, np.inf)
        return inputs_data

    def create_is_inf_net(self, x_shape, x_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(x_type, x_shape, 'x')
            tf.raw_ops.IsInf(x=x, name='is_inf')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[8], x_type=tf.float32),
        dict(x_shape=[3, 7], x_type=tf.float32),
        dict(x_shape=[4, 3, 2], x_type=tf.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_is_inf_basic(self, params, ie_device, precision, ir_version, temp_dir,
                          use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104855')
        if use_legacy_frontend:
            pytest.skip("IsInf operation is not supported via legacy frontend.")
        self._test(*self.create_is_inf_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
