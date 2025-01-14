# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(54654675)


class TestSoftmax(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'logits:0' in inputs_info, "Test error: inputs_info must contain `logits`"
        logits_shape = inputs_info['logits:0']
        inputs_data = {}
        inputs_data['logits:0'] = rng.uniform(-5.0, 5.0, logits_shape).astype(self.input_type)
        return inputs_data

    def create_softmax_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            logits = tf.compat.v1.placeholder(input_type, input_shape, 'logits')
            tf.raw_ops.Softmax(logits=logits)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[2, 6], [1, 3]])
    @pytest.mark.parametrize('input_type', [np.float16, np.float32, np.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_softmax(self, input_shape, input_type,
                     ie_device, precision, ir_version, temp_dir,
                     use_legacy_frontend):
        custom_eps = None
        if input_type == np.float16:
            custom_eps = 1e-3
        self._test(*self.create_softmax_net(input_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend, custom_eps=custom_eps)
