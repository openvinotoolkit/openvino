# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(345345)


class TestExpm1(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = rng.uniform(-2.0, 2.0, x_shape).astype(self.x_type)
        return inputs_data

    def create_expm1_net(self, x_shape, x_type):
        self.x_type = x_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(x_type, x_shape, 'x')
            tf.raw_ops.Expm1(x=x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('x_shape', [[], [2], [3, 4], [3, 2, 1, 4]])
    @pytest.mark.parametrize('x_type', [np.float16, np.float32, np.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_expm1(self, x_shape, x_type, ie_device, precision, ir_version, temp_dir,
                   use_legacy_frontend):
        self._test(*self.create_expm1_net(x_shape, x_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
