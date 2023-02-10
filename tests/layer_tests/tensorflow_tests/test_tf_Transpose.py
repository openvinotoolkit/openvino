# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestTranspose(CommonTFLayerTest):
    def create_transpose_net(self, x_shape, perm_value):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            perm = tf.constant(perm_value, dtype=tf.int32)
            tf.raw_ops.Transpose(x=x, perm=perm)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[2, 4], perm_value=[1, 0]),
        dict(x_shape=[2, 1, 3, 4], perm_value=[2, 0, 1, 3]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_transpose_basic(self, params, ie_device, precision, ir_version, temp_dir,
                             use_new_frontend, use_old_api):
        self._test(*self.create_transpose_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
