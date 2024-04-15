# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasDepthwiseConv2D(CommonTF2LayerTest):
    def create_keras_dconv2D_net(self, input_names, input_shapes, input_type, kernel_size, strides,
                                 padding,
                                 depth_multiplier, data_format, dilation_rate, activation, use_bias,
                                 ir_version):
        """
               create TensorFlow 2 model with Keras DeptwiseConv2D operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                                            padding=padding,
                                            depth_multiplier=depth_multiplier,
                                            data_format=data_format,
                                            dilation_rate=dilation_rate, activation=activation,
                                            use_bias=use_bias)(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None

        return tf2_net, ref_net

    test_data_format_padding = [
        dict(input_names=["x"], input_shapes=[[5, 7, 16, 3]], input_type=tf.float32, kernel_size=1,
             strides=1, padding='valid', depth_multiplier=2, data_format='channels_last',
             dilation_rate=2, activation=None, use_bias=False),
        dict(input_names=["x"], input_shapes=[[5, 16, 4, 4]], input_type=tf.float32,
             kernel_size=(3, 3), strides=(4, 4), padding='valid', depth_multiplier=2,
             data_format='channels_first', dilation_rate=1, activation=None, use_bias=False),
        dict(input_names=["x"], input_shapes=[[5, 8, 16, 4]], input_type=tf.float32,
             kernel_size=(2, 2), strides=1, padding='same', depth_multiplier=2,
             data_format='channels_last',
             dilation_rate=(2, 2), activation=None, use_bias=False),
        dict(input_names=["x"], input_shapes=[[5, 16, 8, 4]], input_type=tf.float32,
             kernel_size=(2, 2), strides=(3, 3), padding='same', depth_multiplier=4,
             data_format='channels_first',
             dilation_rate=1, activation=None, use_bias=False),
    ]

    @pytest.mark.parametrize("params", test_data_format_padding)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_dconv2D_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        self._test(*self.create_keras_dconv2D_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_use_bias_true = [
        dict(input_names=["x"], input_shapes=[[5, 7, 3, 3]], input_type=tf.float32, kernel_size=1,
             strides=1, padding='valid', depth_multiplier=2, data_format='channels_last',
             dilation_rate=2, activation=None, use_bias=True),
        dict(input_names=["x"], input_shapes=[[5, 16, 16, 4]], input_type=tf.float32,
             kernel_size=(3, 3), strides=(4, 4), padding='valid', depth_multiplier=2,
             data_format='channels_first', dilation_rate=1, activation=None, use_bias=True),
        dict(input_names=["x"], input_shapes=[[5, 8, 16, 4]], input_type=tf.float32,
             kernel_size=(2, 2), strides=1, padding='same', depth_multiplier=2,
             data_format='channels_last',
             dilation_rate=(2, 2), activation=None, use_bias=True),
        dict(input_names=["x"], input_shapes=[[5, 16, 8, 4]], input_type=tf.float32,
             kernel_size=(2, 2), strides=(3, 3), padding='same', depth_multiplier=4,
             data_format='channels_first',
             dilation_rate=1, activation=None, use_bias=True),
    ]

    @pytest.mark.parametrize("params", test_data_use_bias_true)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_use_bias_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                    use_legacy_frontend):
        self._test(*self.create_keras_dconv2D_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_activations = [
        dict(input_names=["x"], input_shapes=[[5, 7, 16, 3]], input_type=tf.float32, kernel_size=1,
             strides=1, padding='valid', depth_multiplier=2, data_format='channels_last',
             dilation_rate=2, activation='softmax', use_bias=True),
        dict(input_names=["x"], input_shapes=[[5, 16, 16, 4]], input_type=tf.float32,
             kernel_size=(3, 3), strides=(4, 4), padding='valid', depth_multiplier=2,
             data_format='channels_first', dilation_rate=1, activation='elu', use_bias=True),
        dict(input_names=["x"], input_shapes=[[5, 8, 16, 4]], input_type=tf.float32,
             kernel_size=(2, 2), strides=1, padding='same', depth_multiplier=2,
             data_format='channels_last',
             dilation_rate=(2, 2), activation='linear', use_bias=True),
        dict(input_names=["x"], input_shapes=[[5, 16, 8, 4]], input_type=tf.float32,
             kernel_size=(2, 2), strides=(3, 3), padding='same', depth_multiplier=4,
             data_format='channels_first',
             dilation_rate=1, activation='tanh', use_bias=True),
    ]

    @pytest.mark.parametrize("params", test_data_activations)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_activations_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                       use_legacy_frontend):
        self._test(*self.create_keras_dconv2D_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
