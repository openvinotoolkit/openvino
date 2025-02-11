# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestLRN(CommonTFLayerTest):
    def create_lrn_net(self, input_shape, depth_radius, bias, alpha, beta):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
            tf.raw_ops.LRN(input=input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 20, 30, 3], depth_radius=3, bias=0.2, alpha=4, beta=0.75),
        dict(input_shape=[1, 10, 30, 2], depth_radius=4, bias=1, alpha=3, beta=0.15),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    #@pytest.mark.precommit - ticket 116032
    @pytest.mark.nightly
    def test_lrn_basic(self, params, ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        self._test(*self.create_lrn_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
