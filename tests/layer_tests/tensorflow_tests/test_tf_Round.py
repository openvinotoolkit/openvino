# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
import numpy as np
from common.tf_layer_test_class import CommonTFLayerTest

class TestTFRound(CommonTFLayerTest):
    def create_tf_round_net(self, input_shape, input_type):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'Input')
            round = tf.raw_ops.Round(x=input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data_basic = [
            dict(input_shape=[6], input_type=tf.float32),
            dict(input_shape=[2, 5, 3], input_type=tf.int32),
            dict(input_shape=[10, 5, 1, 5], input_type=tf.float32),
        ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_round_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_tf_round_net(**params),
                ie_device, precision, ir_version, temp_dir=temp_dir,
                use_legacy_frontend=use_legacy_frontend)