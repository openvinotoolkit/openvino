# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest

import logging

# Testing operation Conv2DBackpropInput
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Conv2DBackpropInput

class TestConv2DBackprop(CommonTFLayerTest):
    # input_shape - should be an array, shape of input tensor in format [batch, height, width, channels]
    # input_filter - should be an array, defines a filter
    # input_strides - should be an array, defines strides of a sliding window to use
    # input_padding - should be a string, defines padding algorithm
    # ir_version - common parameter
    # use_new_frontend - common parameter
    def create_conv2dbackprop_placeholder_const_net(self, input_shape, input_filter, out_backprop, input_strides, input_padding, dilations, ir_version, use_new_frontend):
        """
            TensorFlow net                               IR net

            Placeholder->Conv2DBackpropInput    =>       Placeholder->Transpose->ConvolutionBackpropData->Transpose
                         /                                                       /
            Placeholder-/                                Placeholder->Transpose-/

        """

        import tensorflow as tf

        if dilations is None:
            dilations = [1, 1, 1, 1] #default value regarding Documentation
        else:
            pytest.skip('Dilations != 1 isn\' supported on CPU by TensorFlow')

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.constant(input_shape)
            tf_filter = tf.compat.v1.placeholder(tf.float32, input_filter, "Input")
            tf_backprop = tf.compat.v1.placeholder(tf.float32, out_backprop, "InputBack")

            tf.raw_ops.Conv2DBackpropInput(input_sizes = tf_input, filter = tf_filter, out_backprop = tf_backprop, strides = input_strides, padding = input_padding, dilations = dilations)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shape=[1, 10, 10, 1], input_filter=[1, 1, 1, 1], out_backprop=[1, 10, 10, 1], input_strides=[1, 1, 1, 1], input_padding='SAME', dilations=None),
        dict(input_shape=[1, 10, 10, 3], input_filter=[2, 2, 3, 3], out_backprop=[1, 10, 10, 3], input_strides=[1, 1, 1, 1], input_padding='SAME', dilations=None),
        dict(input_shape=[1, 10, 10, 3], input_filter=[2, 2, 3, 3], out_backprop=[1, 5, 5, 3], input_strides=[1, 2, 2, 1], input_padding='SAME', dilations=None),
        dict(input_shape=[1, 10, 10, 1], input_filter=[1, 1, 1, 1], out_backprop=[1, 10, 10, 1], input_strides=[1, 1, 1, 1], input_padding='VALID', dilations=None),
        dict(input_shape=[1, 10, 10, 3], input_filter=[2, 2, 3, 3], out_backprop=[1, 9, 9, 3], input_strides=[1, 1, 1, 1], input_padding='VALID', dilations=None),
        dict(input_shape=[1, 10, 10, 3], input_filter=[2, 2, 3, 3], out_backprop=[1, 5, 5, 3], input_strides=[1, 2, 2, 1], input_padding='VALID', dilations=None),
        pytest.param(
            dict(input_shape=[1, 56, 56, 3], input_filter=[2, 3, 3, 3], out_backprop=[1, 28, 28, 3], input_strides=[1, 2, 2, 1], input_padding='SAME', dilations=None),
            marks=pytest.mark.precommit_tf_fe),
        pytest.param(
            dict(input_shape=[1, 64, 48, 3], input_filter=[3, 2, 3, 3], out_backprop=[1, 31, 24, 3], input_strides=[1, 2, 2, 1], input_padding='VALID', dilations=None),
            marks=pytest.mark.precommit_tf_fe),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_conv2dbackprop_placeholder_const(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend):
        self._test(*self.create_conv2dbackprop_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
