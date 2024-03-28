# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestReverseV2(CommonTFLayerTest):
    def create_reverse_v2_net(self, shape, axis):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape, 'Input')
            tf.raw_ops.ReverseV2(tensor=x, axis=axis, name='reverse')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(shape=[5], axis=[0]),
        dict(shape=[3], axis=[-1]),
        dict(shape=[2, 3], axis=[1]),
        dict(shape=[2, 3, 5], axis=[-2]),
        dict(shape=[2, 3, 5, 7], axis=[3]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reverse_v2_basic(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reverse_v2_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
