# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestL2Loss(CommonTFLayerTest):
    def create_l2_loss_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
            tf.raw_ops.L2Loss(t=input, name='l2_loss')
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1, 2]),
        dict(input_shape=[2, 3, 4]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_l2_loss_basic(self, params, ie_device, precision, ir_version, temp_dir,
                           use_new_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104863')
        if not use_new_frontend:
            pytest.skip("L2Loss is not supported by legacy FE.")
        self._test(*self.create_l2_loss_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
