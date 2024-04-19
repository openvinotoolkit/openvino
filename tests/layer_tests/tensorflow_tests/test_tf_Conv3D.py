# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


# Testing operation Conv3D
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Conv3D

class TestConv3D(CommonTFLayerTest):
    # input_shape - should be an array, shape of input tensor in format [batch, depth, height, width, channels]
    # input_filter - should be an array, defines a filter
    # input_strides - should be an array, defines strides of a sliding window to use
    # input_padding - should be a string, defines padding algorithm
    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    def create_conv3d_placeholder_const_net(self, input_shape, input_filter, input_strides, input_padding, dilations,
                                            ir_version, use_legacy_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->Conv3D    =>       Placeholder->Transpose->Convolution->Transpose
                         /                                          /
            Placeholder-/                   Placeholder->Transpose-/

        """

        import tensorflow as tf

        if dilations is None:
            dilations = [1, 1, 1, 1, 1]  # default value regarding Documentation
        else:
            pytest.skip('Dilations != 1 isn\' supported on CPU by Tensorflow')

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.compat.v1.placeholder(tf.float32, input_shape, "InputShape")
            tf_filter = tf.compat.v1.placeholder(tf.float32, input_filter, "InputFilter")

            tf.raw_ops.Conv3D(input=tf_input, filter=tf_filter, strides=input_strides, padding=input_padding,
                              dilations=dilations)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shape=[1, 10, 10, 10, 1], input_filter=[3, 3, 3, 1, 1], input_strides=[1, 1, 1, 1, 1],
             dilations=None),
        dict(input_shape=[1, 10, 10, 10, 2], input_filter=[3, 3, 3, 2, 2], input_strides=[1, 1, 1, 1, 1],
             dilations=None),
        dict(input_shape=[1, 10, 10, 10, 2], input_filter=[3, 3, 3, 2, 1], input_strides=[1, 1, 1, 1, 1],
             dilations=None),
        dict(input_shape=[1, 16, 16, 16, 3], input_filter=[4, 2, 4, 3, 3], input_strides=[1, 2, 2, 2, 1],
             dilations=None),
        dict(input_shape=[1, 10, 10, 20, 3], input_filter=[2, 4, 2, 3, 3], input_strides=[1, 2, 2, 2, 1],
             dilations=None),
        dict(input_shape=[1, 10, 10, 10, 4], input_filter=[3, 3, 3, 4, 2], input_strides=[1, 1, 1, 1, 1],
             dilations=None),
        dict(input_shape=[1, 10, 10, 20, 3], input_filter=[2, 4, 2, 3, 3], input_strides=[1, 2, 2, 2, 1],
             dilations=[1, 2, 2, 2, 1]),
        pytest.param(
            dict(input_shape=[1, 24, 24, 24, 3], input_filter=[1, 2, 3, 3, 2], input_strides=[1, 2, 2, 2, 1],
                 dilations=None),
            marks=pytest.mark.precommit),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("padding", ['SAME', 'VALID'])
    @pytest.mark.nightly
    def test_conv3d_placeholder_const(self, params, padding, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("Unable to convert data format b_fs_zyx_fsv16 to weights format issue on GPU")
        self._test(*self.create_conv3d_placeholder_const_net(**params, input_padding=padding, ir_version=ir_version,
                                                             use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, input_padding=padding, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
