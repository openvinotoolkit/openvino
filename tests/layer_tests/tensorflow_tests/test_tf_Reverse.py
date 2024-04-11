# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestReverse(CommonTFLayerTest):
    def create_reverse_net(self, shape, dims):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape, 'Input')
            tf.raw_ops.Reverse(tensor=x, dims=dims, name='reverse')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(shape=[4], dims=[True]),
        dict(shape=[3, 2], dims=[False, True]),
        dict(shape=[4, 2, 3], dims=[False, True, False]),
        dict(shape=[1, 2, 4, 3], dims=[True, False, False, False]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_reverse_basic(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU' and params['shape'] == [1, 2, 4, 3]:
            pytest.skip("No layout format available for reversesequence:ReverseSequence_8462418 issue on GPU")
        self._test(*self.create_reverse_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
