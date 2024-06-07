# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


# Testing operation Conv2D
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Conv2D

class TestConv2D(CommonTFLayerTest):
    # input_shape - should be an array, shape of input tensor in format [batch, height, width, channels]
    # input_filter - should be an array, defines a filter
    # input_strides - should be an array, defines strides of a sliding window to use
    # input_padding - should be a string, defines padding algorithm
    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    def create_conv2d_placeholder_const_net(self, input_shape, input_filter, input_strides, input_padding, dilations,
                                            ir_version, use_legacy_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->Conv2D    =>       Placeholder->Transpose->Convolution->Transpose
                         /                                          /
            Placeholder-/                   Placeholder->Transpose-/

        """

        import tensorflow as tf

        if dilations is None:
            dilations = [1, 1, 1, 1]  # default value regarding Documentation

        #               Batch   Height Width  Channel
        expl_paddings = [0, 0, 1, 1, 1, 1, 0, 0]

        if input_padding == 'EXPLICIT' and use_legacy_frontend == False:
            pytest.xfail(reason="100300")

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.compat.v1.placeholder(tf.float32, input_shape, "InputShape")
            tf_filter = tf.compat.v1.placeholder(tf.float32, input_filter, "InputFilter")

            if input_padding != 'EXPLICIT':
                tf.raw_ops.Conv2D(input=tf_input, filter=tf_filter, strides=input_strides, padding=input_padding,
                                  dilations=dilations)
            else:
                tf.raw_ops.Conv2D(input=tf_input, filter=tf_filter, strides=input_strides, padding=input_padding,
                                  explicit_paddings=expl_paddings, dilations=dilations)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shape=[1, 10, 10, 1], input_filter=[3, 3, 1, 1], input_strides=[1, 1, 1, 1], dilations=None),
        dict(input_shape=[1, 10, 10, 2], input_filter=[3, 3, 2, 2], input_strides=[1, 1, 1, 1], dilations=None),
        dict(input_shape=[1, 10, 10, 2], input_filter=[3, 3, 2, 1], input_strides=[1, 1, 1, 1], dilations=None),
        dict(input_shape=[1, 10, 10, 3], input_filter=[2, 2, 3, 3], input_strides=[1, 1, 1, 1], dilations=None),
        dict(input_shape=[1, 16, 16, 3], input_filter=[2, 2, 3, 3], input_strides=[1, 2, 2, 1], dilations=None),
        dict(input_shape=[1, 10, 10, 4], input_filter=[2, 2, 4, 2], input_strides=[1, 1, 1, 1], dilations=None),
        dict(input_shape=[1, 16, 16, 3], input_filter=[2, 2, 3, 3], input_strides=[1, 2, 2, 1], dilations=[1, 2, 2, 1]),
        pytest.param(
            dict(input_shape=[1, 224, 224, 3], input_filter=[4, 4, 3, 2], input_strides=[1, 2, 2, 1],
                 dilations=[1, 2, 2, 1]),
            marks=pytest.mark.precommit),
        # with four groups
        pytest.param(
            dict(input_shape=[2, 224, 224, 4], input_filter=[4, 4, 1, 12], input_strides=[1, 2, 2, 1],
                 dilations=[1, 2, 2, 1]),
            marks=pytest.mark.precommit)
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("padding", ['EXPLICIT', 'SAME', 'VALID'])
    @pytest.mark.nightly
    def test_conv2d_placeholder_const(self, params, padding, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104862')
        self._test(*self.create_conv2d_placeholder_const_net(**params, input_padding=padding, ir_version=ir_version,
                                                             use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, input_padding=padding, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
