# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(475912)


class TestConv2DBackpropInput(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'filter:0' in inputs_info, "Test error: inputs_info must contain `filter`"
        assert 'out_backprop:0' in inputs_info, "Test error: inputs_info must contain `out_backprop`"

        filter_shape = inputs_info['filter:0']
        out_backprop_shape = inputs_info['out_backprop:0']
        inputs_data = {}
        if np.issubdtype(self.input_type, np.floating):
            inputs_data['filter:0'] = rng.uniform(-1.0, 1.0, filter_shape).astype(self.input_type)
            inputs_data['out_backprop:0'] = rng.uniform(-1.0, 1.0, out_backprop_shape).astype(self.input_type)
        return inputs_data

    def create_conv2d_backprop_input_net(self, input_sizes, filter_shape, out_backprop_shape, strides,
                                         padding, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_sizes = tf.constant(input_sizes, dtype=tf.int32)
            filter = tf.compat.v1.placeholder(input_type, filter_shape, "filter")
            out_backprop = tf.compat.v1.placeholder(input_type, out_backprop_shape, "out_backprop")

            tf.raw_ops.Conv2DBackpropInput(input_sizes=input_sizes, filter=filter, out_backprop=out_backprop,
                                           strides=strides, padding=padding)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_sizes=[1, 10, 10, 1], filter_shape=[1, 1, 1, 1], out_backprop_shape=[1, 10, 10, 1],
             strides=[1, 1, 1, 1]),
        dict(input_sizes=[1, 10, 10, 3], filter_shape=[2, 2, 3, 3], out_backprop_shape=[1, 5, 5, 3],
             strides=[1, 2, 2, 1]),
        dict(input_sizes=[1, 20, 20, 3], filter_shape=[2, 2, 3, 3], out_backprop_shape=[1, 10, 10, 3],
             strides=[1, 2, 2, 1]),
        dict(input_sizes=[1, 20, 20, 1], filter_shape=[1, 1, 1, 1], out_backprop_shape=[1, 20, 20, 1],
             strides=[1, 1, 1, 1]),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("padding", ['SAME', 'VALID'])
    @pytest.mark.parametrize("input_type", [np.float16, np.float32, np.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_create_conv2d_backprop_input(self, params, padding, input_type,
                                          ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        custom_eps = None
        if input_type == np.float16:
            custom_eps = 2 * 1e-3
        self._test(*self.create_conv2d_backprop_input_net(**params, padding=padding, input_type=input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend, custom_eps=custom_eps)
