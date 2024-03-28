# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestZerosLike(CommonTFLayerTest):
    def create_zeros_like_net(self, x_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            tf.raw_ops.ZerosLike(x=x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[]),
        dict(x_shape=[3]),
        dict(x_shape=[2, 1, 4]),
        dict(x_shape=[2, 4, 3, 1]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_zeros_like_basic(self, params, ie_device, precision, ir_version, temp_dir,
                              use_legacy_frontend):
        self._test(*self.create_zeros_like_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
