# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSoftmax(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'logits' in inputs_info
        logits_shape = inputs_info['logits']
        inputs_data = {}
        inputs_data['logits'] = np.random.randint(-5, 5, logits_shape).astype(np.float32)

        return inputs_data

    def create_softmax_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            logits = tf.compat.v1.placeholder(tf.float32, input_shape, 'logits')
            tf.raw_ops.Softmax(logits=logits)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 6]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_softmax_basic(self, params, ie_device, precision, ir_version, temp_dir,
                           use_new_frontend, use_old_api):
        self._test(*self.create_softmax_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
