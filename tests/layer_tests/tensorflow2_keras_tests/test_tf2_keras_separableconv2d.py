# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasSeparableConv2D(CommonTF2LayerTest):
    def create_keras_separableconv2d_net(self, input_names, input_shapes, input_type,
                                         filters, kernel_size, strides, padding, data_format,
                                         dilation_rate, depth_multiplier, activation, use_bias,
                                         ir_version):
        # create TensorFlow 2 model with Keras SeparableConv2D operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.SeparableConv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate, depth_multiplier=depth_multiplier,
            activation=activation, use_bias=use_bias)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    # Tests for different filters number, kernel size, strides, and dilation
    test_data = [
        dict(input_names=["x1"], input_shapes=[[5, 14, 10, 3]], input_type=tf.float32,
             filters=3, kernel_size=(2), strides=1, padding='valid', data_format='channels_last',
             dilation_rate=1, depth_multiplier=1, activation='relu', use_bias=True),
        pytest.param(
            dict(input_names=["x1"], input_shapes=[[1, 10, 16, 2]], input_type=tf.float32,
                 filters=4, kernel_size=(3, 2), strides=3, padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 2), depth_multiplier=1, activation='relu', use_bias=True),
            marks=pytest.mark.xfail(reason="49539")),
        dict(input_names=["x1"], input_shapes=[[2, 18, 15, 3]], input_type=tf.float32,
             filters=2, kernel_size=(1, 3), strides=1, padding='valid', data_format='channels_last',
             dilation_rate=4, depth_multiplier=1, activation='relu', use_bias=True),
        dict(input_names=["x1"], input_shapes=[[2, 22, 4, 8]], input_type=tf.float32,
             filters=3, kernel_size=(2, 2), strides=1, padding='valid', data_format='channels_last',
             dilation_rate=(4, 3), depth_multiplier=6, activation='relu', use_bias=True),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_separableconv2d(self, params, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        self._test(*self.create_keras_separableconv2d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    # Tests for different activations
    test_data_different_activations = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 10, 13]], input_type=tf.float32,
             filters=3, kernel_size=(2, 3), strides=1, padding='valid', data_format='channels_last',
             dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True),
        pytest.param(
            dict(input_names=["x1"], input_shapes=[[2, 22, 14, 5]], input_type=tf.float32,
                 filters=3, kernel_size=(2, 1), strides=1, padding='valid',
                 data_format='channels_last',
                 dilation_rate=4, depth_multiplier=6, activation='selu', use_bias=True),
            marks=pytest.mark.xfail(reason="49512")),
        dict(input_names=["x1"], input_shapes=[[1, 10, 16, 2]], input_type=tf.float32,
             filters=4, kernel_size=(3, 4), strides=(3, 3), padding='valid',
             data_format='channels_last',
             dilation_rate=1, depth_multiplier=1, activation='tanh', use_bias=True),
        dict(input_names=["x1"], input_shapes=[[2, 18, 15, 7]], input_type=tf.float32,
             filters=2, kernel_size=(3, 5), strides=1, padding='valid', data_format='channels_last',
             dilation_rate=2, depth_multiplier=1, activation='swish', use_bias=True),
        dict(input_names=["x1"], input_shapes=[[2, 22, 14, 5]], input_type=tf.float32,
             filters=3, kernel_size=(2, 4), strides=1, padding='valid', data_format='channels_last',
             dilation_rate=4, depth_multiplier=6, activation='linear', use_bias=True),
        dict(input_names=["x1"], input_shapes=[[2, 22, 14, 6]], input_type=tf.float32,
             filters=3, kernel_size=(2, 3), strides=(1, 1), padding='valid',
             data_format='channels_last',
             dilation_rate=1, depth_multiplier=6, activation='elu', use_bias=True),
    ]

    @pytest.mark.parametrize("params", test_data_different_activations)
    @pytest.mark.nightly
    def test_keras_separableconv2d_different_activations(self, params, ie_device, precision,
                                                         ir_version, temp_dir,
                                                         use_legacy_frontend):
        self._test(*self.create_keras_separableconv2d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    # Tests for different padding
    test_data_different_padding = [
        dict(input_names=["x1"], input_shapes=[[5, 14, 10, 3]], input_type=tf.float32,
             filters=3, kernel_size=(2), strides=1, padding='valid', data_format='channels_last',
             dilation_rate=1, depth_multiplier=1, activation='relu', use_bias=True),
        dict(input_names=["x1"], input_shapes=[[3, 18, 15, 4]], input_type=tf.float32,
             filters=3, kernel_size=(4, 1), strides=1, padding='same', data_format='channels_last',
             dilation_rate=1, depth_multiplier=1, activation='relu', use_bias=True),
    ]

    @pytest.mark.parametrize("params", test_data_different_padding)
    @pytest.mark.nightly
    def test_keras_separableconv2d_different_padding(self, params, ie_device, precision, ir_version,
                                                     temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_separableconv2d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    # Tests for different bias
    test_data_different_bias = [
        dict(input_names=["x1"], input_shapes=[[5, 17, 14, 3]], input_type=tf.float32,
             filters=3, kernel_size=(2, 1), strides=1, padding='valid', data_format='channels_last',
             dilation_rate=1, depth_multiplier=1, activation='relu', use_bias=True),
        dict(input_names=["x1"], input_shapes=[[1, 14, 12, 2]], input_type=tf.float32,
             filters=4, kernel_size=(2, 2), strides=1, padding='valid', data_format='channels_last',
             dilation_rate=2, depth_multiplier=3, activation='relu', use_bias=False),
    ]

    @pytest.mark.parametrize("params", test_data_different_bias)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_keras_separableconv2d_different_bias(self, params, ie_device, precision, ir_version,
                                                  temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_separableconv2d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
