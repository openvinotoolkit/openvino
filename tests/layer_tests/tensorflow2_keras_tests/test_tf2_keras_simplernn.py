# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasSimpleRNN(CommonTF2LayerTest):
    def create_keras_simplernn_net(self, input_names, input_shapes, input_type,
                                   units, activation, use_bias, dropout, recurrent_dropout,
                                   return_sequences,
                                   return_state, go_backwards, stateful, unroll,
                                   ir_version):
        # create TensorFlow 2 model with Keras SimpleRNN operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.SimpleRNN(units=units, activation=activation, use_bias=use_bias,
                                      dropout=dropout,
                                      recurrent_dropout=recurrent_dropout,
                                      return_sequences=return_sequences,
                                      return_state=return_state, go_backwards=go_backwards,
                                      stateful=stateful,
                                      unroll=unroll)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    # Tests for different activation functions
    test_data_different_activations = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 2]], input_type=tf.float32,
             units=3, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0,
             return_sequences=False,
             return_state=False, go_backwards=False, stateful=False, unroll=False),
        dict(input_names=["x1"], input_shapes=[[2, 3, 6]], input_type=tf.float32,
             units=3, activation='relu', use_bias=True, dropout=0.0, recurrent_dropout=0.0,
             return_sequences=False,
             return_state=False, go_backwards=False, stateful=False, unroll=False),
        dict(input_names=["x1"], input_shapes=[[3, 4, 1]], input_type=tf.float32,
             units=3, activation='elu', use_bias=True, dropout=0.0, recurrent_dropout=0.0,
             return_sequences=False,
             return_state=False, go_backwards=False, stateful=False, unroll=False),
        dict(input_names=["x1"], input_shapes=[[5, 1, 3]], input_type=tf.float32,
             units=3, activation='selu', use_bias=True, dropout=0.0, recurrent_dropout=0.0,
             return_sequences=False,
             return_state=False, go_backwards=False, stateful=False, unroll=False),
    ]

    @pytest.mark.parametrize("params", test_data_different_activations)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_simplernn_different_activations(self, params, ie_device, precision, ir_version,
                                                   temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_simplernn_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    # Tests for RNN with dropout
    test_data_dropout = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 2]], input_type=tf.float32,
             units=3, activation='tanh', use_bias=True, dropout=0.8, recurrent_dropout=0.0,
             return_sequences=False,
             return_state=False, go_backwards=False, stateful=False, unroll=False),
        dict(input_names=["x1"], input_shapes=[[5, 4, 2]], input_type=tf.float32,
             units=3, activation='tanh', use_bias=True, dropout=0.8, recurrent_dropout=0.3,
             return_sequences=False,
             return_state=False, go_backwards=False, stateful=False, unroll=False),
    ]

    @pytest.mark.parametrize("params", test_data_dropout)
    @pytest.mark.nightly
    def test_keras_simplernn_dropout(self, params, ie_device, precision, ir_version, temp_dir,
                                     use_legacy_frontend):
        self._test(*self.create_keras_simplernn_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    # Tests for RNN with other attributes
    test_data_other = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 2]], input_type=tf.float32,
             units=3, activation='tanh', use_bias=False, dropout=0.0, recurrent_dropout=0.0,
             return_sequences=False,
             return_state=False, go_backwards=False, stateful=False, unroll=False),
        dict(input_names=["x1"], input_shapes=[[5, 4, 2]], input_type=tf.float32,
             units=3, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0,
             return_sequences=False,
             return_state=False, go_backwards=True, stateful=False, unroll=False),
    ]

    @pytest.mark.parametrize("params", test_data_other)
    @pytest.mark.nightly
    def test_keras_simplernn_other(self, params, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        self._test(*self.create_keras_simplernn_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    # Tests for RNN with multiple outputs
    test_data_multipleoutput = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 2]], input_type=tf.float32,
             units=3, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0,
             return_sequences=True, return_state=False,
             go_backwards=False, stateful=False, unroll=False),
        pytest.param(
            dict(input_names=["x1"], input_shapes=[[5, 4, 2]], input_type=tf.float32,
                 units=3, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=False, return_state=True,
                 go_backwards=False, stateful=False, unroll=False),
            marks=pytest.mark.xfail(reason="49537")),
        pytest.param(
            dict(input_names=["x1"], input_shapes=[[5, 4, 2]], input_type=tf.float32,
                 units=3, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True, return_state=True,
                 go_backwards=False, stateful=False, unroll=False),
            marks=pytest.mark.xfail(reason="49537")),
    ]

    @pytest.mark.parametrize("params", test_data_multipleoutput)
    @pytest.mark.nightly
    def test_keras_simplernn_test_data_multipleoutput(self, params, ie_device, precision,
                                                      ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_simplernn_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
