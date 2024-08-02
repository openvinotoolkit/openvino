# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestLog1p(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-0.9, 5, x_shape).astype(np.float32)

        return inputs_data

    def create_log1p_net(self, x_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            tf.raw_ops.Log1p(x=x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[]),
        dict(x_shape=[3]),
        dict(x_shape=[2, 1, 4]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_log1p_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        self._test(*self.create_log1p_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
