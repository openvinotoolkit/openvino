# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


# Testing operation Conv3DBackpropInputV2
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Conv3DBackpropInputV2

class TestConv3DBackprop(CommonTFLayerTest):
    # input_shape - should be an array, shape of input tensor in format [batch, depth, height, width, channels]
    # input_filter - should be an array, defines a filter
    # out_backprop - should be an array, shape of backprop
    # input_strides - should be an array, defines strides of a sliding window to use
    # input_padding - should be a string, defines padding algorithm
    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    def create_conv3dbackprop_placeholder_const_net(self, input_shape, input_filter, out_backprop, input_strides,
                                                    input_padding, dilations, ir_version, use_legacy_frontend):
        """
            TensorFlow net                                 IR net

            Placeholder->Conv3DBackpropInputV2    =>       Placeholder->Transpose->ConvolutionBackpropData->Transpose
                         /                                                         /
            Placeholder-/                                  Placeholder->Transpose-/
        """

        import tensorflow as tf

        if dilations is None:
            dilations = [1, 1, 1, 1, 1]  # default value regarding Documentation
        else:
            pytest.skip('Dilations != 1 isn\' supported on CPU by TensorFlow')

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.constant(input_shape)
            tf_filter = tf.compat.v1.placeholder(tf.float32, input_filter, "InputFilter")
            tf_backprop = tf.compat.v1.placeholder(tf.float32, out_backprop, "InputBackprop")

            tf.raw_ops.Conv3DBackpropInputV2(input_sizes=tf_input, filter=tf_filter, out_backprop=tf_backprop,
                                             strides=input_strides, padding=input_padding)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shape=[1, 15, 15, 15, 1], input_filter=[1, 1, 1, 1, 1], out_backprop=[1, 15, 15, 15, 1],
             input_strides=[1, 1, 1, 1, 1], input_padding='SAME', dilations=None),
        dict(input_shape=[1, 15, 15, 15, 2], input_filter=[1, 1, 1, 2, 2], out_backprop=[1, 15, 15, 15, 2],
             input_strides=[1, 1, 1, 1, 1], input_padding='SAME', dilations=None),
        dict(input_shape=[1, 16, 16, 16, 3], input_filter=[2, 2, 2, 3, 3], out_backprop=[1, 8, 8, 8, 3],
             input_strides=[1, 2, 2, 2, 1], input_padding='SAME', dilations=None),
        dict(input_shape=[1, 10, 10, 20, 3], input_filter=[2, 2, 2, 3, 3], out_backprop=[1, 5, 5, 10, 3],
             input_strides=[1, 2, 2, 2, 1], input_padding='SAME', dilations=None),
        dict(input_shape=[1, 15, 15, 15, 1], input_filter=[1, 1, 1, 1, 1], out_backprop=[1, 15, 15, 15, 1],
             input_strides=[1, 1, 1, 1, 1], input_padding='VALID', dilations=None),
        dict(input_shape=[1, 15, 15, 15, 2], input_filter=[1, 1, 1, 2, 2], out_backprop=[1, 15, 15, 15, 2],
             input_strides=[1, 1, 1, 1, 1], input_padding='VALID', dilations=None),
        dict(input_shape=[1, 16, 16, 16, 3], input_filter=[2, 2, 2, 3, 3], out_backprop=[1, 8, 8, 8, 3],
             input_strides=[1, 2, 2, 2, 1], input_padding='VALID', dilations=None),
        dict(input_shape=[1, 10, 10, 20, 3], input_filter=[2, 2, 2, 3, 3], out_backprop=[1, 5, 5, 10, 3],
             input_strides=[1, 2, 2, 2, 1], input_padding='VALID', dilations=None),
        pytest.param(
            dict(input_shape=[1, 16, 20, 10, 3], input_filter=[3, 2, 4, 3, 3], out_backprop=[1, 8, 10, 5, 3],
                 input_strides=[1, 2, 2, 2, 1], input_padding='SAME', dilations=None),
            marks=pytest.mark.precommit),
        pytest.param(
            dict(input_shape=[1, 16, 16, 16, 3], input_filter=[4, 2, 3, 3, 3], out_backprop=[1, 7, 8, 7, 3],
                 input_strides=[1, 2, 2, 2, 1], input_padding='VALID', dilations=None),
            marks=pytest.mark.precommit),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_conv3dbackprop_placeholder_const(self, params, ie_device, precision, ir_version, temp_dir,
                                              use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("Unable to convert data format b_fs_zyx_fsv16 to weights format issue on GPU")
        self._test(*self.create_conv3dbackprop_placeholder_const_net(**params, ir_version=ir_version,
                                                                     use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
