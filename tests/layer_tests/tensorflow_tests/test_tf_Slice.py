# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSlice(CommonTFLayerTest):
    def create_slice_net(self, input_shape, input_type, begin_value, size_value):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_x = tf.compat.v1.placeholder(input_type, input_shape, 'input_x')
            begin = tf.constant(begin_value, tf.int32)
            size = tf.constant(size_value, tf.int32)
            tf.raw_ops.Slice(input=input_x, begin=begin, size=size)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data_basic = [
        dict(input_shape=[6], input_type=tf.float32, begin_value=[2], size_value=[2]),
        dict(input_shape=[2, 5, 3], input_type=tf.int32, begin_value=[0, 1, 0], size_value=[-1, 1, -1]),
        dict(input_shape=[10, 5, 1, 5], input_type=tf.float32, begin_value=[5, 1, 0, 3], size_value=[2, 4, -1, -1]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_slice_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_slice_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
