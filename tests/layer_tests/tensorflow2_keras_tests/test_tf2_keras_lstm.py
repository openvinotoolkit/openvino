# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasLSTM(CommonTF2LayerTest):
    def create_keras_lstm_net(self, input_names, input_shapes, input_type, units, activation,
                              recurrent_activation,
                              use_bias, dropouts, flags, ir_version):
        """
                create TensorFlow 2 model with Keras LSTM operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        # add additional layer Add to avoid
        # double tensor name for Parameter after EliminateLoopInputsOutputs elimination
        x1_post = tf.keras.layers.Lambda(lambda x: x + 0.01)(x1)
        dropout, recurrent_dropout = dropouts
        go_backwards = flags
        y = tf.keras.layers.LSTM(units=units, activation=activation,
                                 recurrent_activation=recurrent_activation,
                                 use_bias=use_bias, dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=False, return_state=False,
                                 go_backwards=go_backwards)(x1_post)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None

        return tf2_net, ref_net

    test_data_simple = [
        dict(input_names=["x"], input_shapes=[[2, 2, 3]], input_type=tf.float32, units=1,
             activation='tanh', recurrent_activation='sigmoid', dropouts=(.0, .3), use_bias=True,
             flags=False),
        dict(input_names=["x"], input_shapes=[[1, 2, 3]], input_type=tf.float32, units=4,
             activation='relu', recurrent_activation='sigmoid', dropouts=(.2, .4), use_bias=True,
             flags=False),
        dict(input_names=["x"], input_shapes=[[3, 2, 3]], input_type=tf.float32, units=2,
             activation='elu', recurrent_activation='tanh', dropouts=(.3, .5), use_bias=True,
             flags=False),
        dict(input_names=["x"], input_shapes=[[2, 3, 4]], input_type=tf.float32, units=1,
             activation='elu', recurrent_activation='softmax', dropouts=(.0, .5), use_bias=True,
             flags=False),
        dict(input_names=["x"], input_shapes=[[1, 3, 4]], input_type=tf.float32, units=3,
             activation='linear', recurrent_activation='sigmoid', dropouts=(.4, .6),
             flags=False, use_bias=True),
    ]

    @pytest.mark.parametrize("params", test_data_simple)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_lstm_with_bias_float32(self, params, ie_device, precision, temp_dir, ir_version,
                                          use_legacy_frontend):
        self._test(*self.create_keras_lstm_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_without_bias = [
        dict(input_names=["x"], input_shapes=[[2, 2, 7]], input_type=tf.float32, units=1,
             activation='tanh',
             recurrent_activation='sigmoid', dropouts=(.0, .3), use_bias=False,
             flags=False),
        dict(input_names=["x"], input_shapes=[[3, 8, 3]], input_type=tf.float32, units=4,
             activation='relu',
             recurrent_activation='sigmoid', dropouts=(.7, .4), use_bias=False,
             flags=False),
        dict(input_names=["x"], input_shapes=[[4, 2, 2]], input_type=tf.float32, units=2,
             activation='elu',
             recurrent_activation='tanh', dropouts=(.0, .5), use_bias=False, flags=False),
    ]

    @pytest.mark.parametrize("params", test_data_without_bias)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_lstm_without_bias_float32(self, params, ie_device, precision, temp_dir,
                                             ir_version, use_legacy_frontend):
        self._test(*self.create_keras_lstm_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_different_flags = [
        dict(input_names=["x"], input_shapes=[[2, 3, 2]], input_type=tf.float32, units=1,
             activation='elu',
             recurrent_activation='sigmoid', dropouts=(.0, .3), use_bias=True, flags=False),
        dict(input_names=["x"], input_shapes=[[4, 8, 3]], input_type=tf.float32, dropouts=(.1, .3),
             units=3,
             activation='relu', use_bias=False, recurrent_activation='tanh', flags=True),
        dict(input_names=["x"], input_shapes=[[4, 2, 7]], input_type=tf.float32, units=5,
             activation='relu',
             recurrent_activation='tanh', dropouts=(.2, .6), use_bias=True, flags=True),
    ]

    @pytest.mark.parametrize("params", test_data_different_flags)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_lstm_flags_float32(self, params, ie_device, precision, temp_dir, ir_version,
                                      use_legacy_frontend):
        if platform.machine() in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']:
            pytest.skip("inference mismatch issue on ARM")
        self._test(*self.create_keras_lstm_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
