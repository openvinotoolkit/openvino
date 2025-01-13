# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(233453)


class TestL2Loss(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info, "Test error: inputs_info must contain `input`"
        input_shape = inputs_info['input:0']
        inputs_data = {}
        inputs_data['input:0'] = rng.uniform(-2.0, 2.0, input_shape).astype(self.input_type)
        return inputs_data

    def create_l2_loss_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.L2Loss(t=input, name='l2_loss')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("input_shape", [[], [2], [1, 2], [2, 3, 4]])
    @pytest.mark.parametrize("input_type", [np.float16, np.float32, np.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_l2_loss_basic(self, input_shape, input_type,
                           ie_device, precision, ir_version, temp_dir,
                           use_legacy_frontend):
        custom_eps = None
        if input_type == np.float16:
            custom_eps = 3 * 1e-3
        if ie_device == 'GPU' and input_shape == []:
            pytest.skip("150321: Accessing out-of-range dimension on GPU")
        self._test(*self.create_l2_loss_net(input_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend, custom_eps=custom_eps)
