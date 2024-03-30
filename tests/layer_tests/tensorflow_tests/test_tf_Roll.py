# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestTFRoll(CommonTFLayerTest):
    def create_tf_roll_net(self, shift, axis, x_shape, input_type, ir_version, use_legacy_frontend):
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, x_shape, 'Input')
            roll = tf.roll(x, shift=shift, axis=axis)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf_net, ref_net

    test_data = [dict(shift=[1], axis=[-1], x_shape=[4, 3], input_type=tf.float32),
                 pytest.param(dict(shift=[1, 5, -7], axis=[0, 1, 1], x_shape=[2, 3, 5], input_type=tf.float16),
                              marks=pytest.mark.precommit),
                 dict(shift=[11, -8], axis=[-1, -2], x_shape=[3, 4, 3, 1], input_type=tf.int32),
                 dict(shift=[7, -2, 5], axis=[0, -1, -1], x_shape=[5, 2, 3, 7],
                      input_type=tf.int64),
                 dict(shift=[3, 7], axis=[0, 1], x_shape=[2, 4, 3, 5, 4], input_type=tf.half),
                 pytest.param(
                     dict(shift=[1, -2], axis=[0, 1], x_shape=[2, 4, 3, 5], input_type=tf.float32),
                     marks=pytest.mark.precommit)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_tf_roll(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("Roll is not supported on GPU")
        self._test(*self.create_tf_roll_net(**params, ir_version=ir_version,
                                            use_legacy_frontend=use_legacy_frontend), ie_device,
                   precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)
