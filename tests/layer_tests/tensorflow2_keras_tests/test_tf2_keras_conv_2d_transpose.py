# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest

rng = np.random.default_rng(233534)


class TestKerasConv2DTranspose(CommonTF2LayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x']
        inputs_data = {}
        inputs_data['x'] = rng.uniform(-2.0, 2.0, x_shape).astype(self.input_type)
        return inputs_data

    def create_keras_conv_2d_transpose_net(self, input_shapes, input_type,
                                           filters, kernel_size,
                                           strides, padding, data_format,
                                           dilation_rate, activation,
                                           use_bias):
        self.input_type = input_type
        activation_func_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # tf.nn.<activation> operation have no "==" operation to be compared
            'relu': tf.nn.relu,
            'sigmoid': tf.nn.sigmoid
        }
        activation = activation_func_structure[activation]

        # create TensorFlow 2 model with Keras Conv2DTranspose operation
        tf.keras.backend.clear_session()
        x = tf.keras.Input(shape=input_shapes[0][1:], dtype=input_type, name='x')

        y = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                            strides=strides, padding=padding, data_format=data_format,
                                            dilation_rate=dilation_rate, activation=activation,
                                            use_bias=use_bias)(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        ref_net = None

        return tf2_net, ref_net

    @pytest.mark.parametrize('input_shapes', [[[3, 9, 7, 8]]])
    @pytest.mark.parametrize('input_type', [np.float32, np.float64])
    @pytest.mark.parametrize('filters', [2, 5])
    @pytest.mark.parametrize('kernel_size', [3, 5])
    @pytest.mark.parametrize('strides', [(1, 2), (2, 2)])
    @pytest.mark.parametrize('padding', ['valid', 'same'])
    @pytest.mark.parametrize('data_format', ['channels_last'])
    @pytest.mark.parametrize('dilation_rate', [(1, 1)])
    @pytest.mark.parametrize('activation', ['sigmoid', 'relu'])
    @pytest.mark.parametrize('use_bias', [True, False])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_keras_conv_2d_transpose(self, input_shapes, input_type, filters, kernel_size,
                                     strides, padding, data_format, dilation_rate, activation,
                                     use_bias,
                                     ie_device, precision,
                                     ir_version, temp_dir, use_legacy_frontend):
        params = {}
        params['input_shapes'] = input_shapes
        self._test(*self.create_keras_conv_2d_transpose_net(input_shapes, input_type,
                                                            filters, kernel_size,
                                                            strides, padding, data_format,
                                                            dilation_rate, activation,
                                                            use_bias),
                   ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
