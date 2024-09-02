# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasConvLSTM2D(CommonTF2LayerTest):
    def _prepare_input(self, inputs_info):
        # Need to generate the input tensor from a range (-1, 1) for ConvLSTM2D
        # to avoid overflows
        input_names = list(inputs_info.keys())
        assert len(input_names) == 1, "Test expects only one input"
        x_shape = inputs_info[input_names[0]]
        inputs_data = {}
        inputs_data[input_names[0]] = np.random.uniform(-1, 1, x_shape).astype(np.float32)

        return inputs_data

    def create_keras_conv_lstm_2d_net(self, params, input_shapes):
        # create TensorFlow 2 model with Keras ConvLSTM2D operation
        tf.keras.backend.clear_session()

        activation = params.get('activation', None)
        recurrent_activation = params.get('recurrent_activation', None)

        if activation is not None:
            params['activation'] = tf.keras.activations.get(activation)
        if recurrent_activation is not None:
            params['recurrent_activation'] = tf.keras.activations.get(recurrent_activation)

        x = tf.keras.Input(shape=input_shapes[0][1:], name="x")
        y = tf.keras.layers.ConvLSTM2D(**params)(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        return tf2_net, None

    test_data_basic = [
        dict(params=dict(filters=4, kernel_size=(3, 3), padding='same', return_sequences=False,
                         activation="swish"),
             input_shapes=[[2, 5, 20, 30, 2]]),
        dict(params=dict(filters=6, kernel_size=(2, 3), padding='valid', dilation_rate=3,
                         recurrent_activation="elu", return_sequences=True, use_bias=True,
                         data_format="channels_last"),
             input_shapes=[[2, 5, 40, 30, 1]]),
        dict(params=dict(filters=3, kernel_size=(3, 3), padding='valid', return_sequences=False),
             input_shapes=[[2, 5, 20, 30, 1]]),
        dict(params=dict(filters=2, kernel_size=(2, 2), padding='same', return_sequences=False, activation="swish"),
             input_shapes=[[2, 5, 25, 15, 3]]),
        dict(params=dict(filters=3, kernel_size=(3, 3), padding='valid', strides=(2, 2),
                         return_sequences=True),
             input_shapes=[[2, 5, 10, 15, 2]]),
        dict(params=dict(filters=5, kernel_size=(2, 2), padding='valid', dilation_rate=3,
                         activation="relu", return_sequences=False, use_bias=True,
                         data_format="channels_last"),
             input_shapes=[[2, 5, 18, 17, 1]])
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_keras_conv_lstm_2d_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_keras_conv_lstm_2d_net(**params), ie_device,
                   precision,
                   temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
