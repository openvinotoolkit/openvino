# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasGru(CommonTF2LayerTest):
    def create_keras_gru_net(self, input_names, input_shapes, input_type, units, activation,
                             recurrent_activation,
                             use_bias, dropouts, flags, ir_version):
        """
                create TensorFlow 2 model with Keras GRU operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        dropout, recurrent_dropout = dropouts
        go_backwards, reset_after = flags
        y = tf.keras.layers.GRU(units=units, activation=activation,
                                recurrent_activation=recurrent_activation,
                                use_bias=use_bias, dropout=dropout,
                                recurrent_dropout=recurrent_dropout,
                                return_sequences=False, return_state=False,
                                go_backwards=go_backwards, reset_after=reset_after)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None

        return tf2_net, ref_net

    test_data_simple = [
        dict(input_names=["x"], input_shapes=[[2, 2, 3]], input_type=tf.float32, units=1,
             activation='tanh', recurrent_activation='sigmoid', dropouts=(.0, .3), use_bias=True,
             flags=(False, False)),
        dict(input_names=["x"], input_shapes=[[1, 2, 3]], input_type=tf.float32, units=4,
             activation='relu', recurrent_activation='sigmoid', dropouts=(.2, .4), use_bias=True,
             flags=(False, False)),
        dict(input_names=["x"], input_shapes=[[3, 2, 3]], input_type=tf.float32, units=2,
             activation='elu', recurrent_activation='tanh', dropouts=(.3, .5), use_bias=True,
             flags=(False, False)),
        dict(input_names=["x"], input_shapes=[[2, 3, 4]], input_type=tf.float32, units=1,
             activation='elu', recurrent_activation='softmax', dropouts=(.0, .5), use_bias=True,
             flags=(False, False)),
        dict(input_names=["x"], input_shapes=[[1, 3, 4]], input_type=tf.float32, units=3,
             activation='linear', recurrent_activation='sigmoid', dropouts=(.4, .6),
             flags=(False, False), use_bias=True)
    ]

    @pytest.mark.parametrize("params", test_data_simple)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_gru_with_bias_float32(self, params, ie_device, precision, temp_dir, ir_version,
                                         use_legacy_frontend):
        self._test(*self.create_keras_gru_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_without_bias = [
        dict(input_names=["x"], input_shapes=[[2, 2, 7]], input_type=tf.float32, units=1,
             activation='tanh', recurrent_activation='sigmoid', dropouts=(.0, .3), use_bias=False,
             flags=(False, False)),
        dict(input_names=["x"], input_shapes=[[3, 8, 3]], input_type=tf.float32, units=4,
             activation='relu', recurrent_activation='sigmoid', dropouts=(.7, .4), use_bias=False,
             flags=(False, False)),
        dict(input_names=["x"], input_shapes=[[4, 2, 2]], input_type=tf.float32, units=2,
             activation='elu', recurrent_activation='tanh', dropouts=(.0, .5), use_bias=False,
             flags=(False, False))
    ]

    @pytest.mark.parametrize("params", test_data_without_bias)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_gru_without_bias_float32(self, params, ie_device, precision, temp_dir,
                                            ir_version, use_legacy_frontend):
        self._test(*self.create_keras_gru_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_different_flags = [
        dict(input_names=["x"], input_shapes=[[2, 3, 2]], input_type=tf.float32, units=1,
             activation='elu', recurrent_activation='sigmoid', dropouts=(.0, .3), use_bias=True,
             flags=(True, False)),
        dict(input_names=["x"], input_shapes=[[4, 8, 3]], input_type=tf.float32, dropouts=(.1, .3),
             units=3, activation='relu', use_bias=False, recurrent_activation='tanh',
             flags=(False, True)),
        dict(input_names=["x"], input_shapes=[[4, 2, 7]], input_type=tf.float32, units=5,
             activation='relu', recurrent_activation='tanh', dropouts=(.2, .6),
             use_bias=True, flags=(False, False)),
        dict(input_names=["x"], input_shapes=[[4, 16, 2]], input_type=tf.float32, units=5,
             activation='relu', recurrent_activation='tanh', dropouts=(.2, .6),
             use_bias=True, flags=(False, True)),
        dict(input_names=["x"], input_shapes=[[4, 8, 7]], input_type=tf.float32, units=5,
             activation='elu', recurrent_activation='sigmoid', dropouts=(.2, .6),
             use_bias=True, flags=(True, True)),
    ]

    @pytest.mark.parametrize("params", test_data_different_flags)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="sporadic inference mismatch")
    def test_keras_gru_flags_float32(self, params, ie_device, precision, temp_dir, ir_version,
                                     use_legacy_frontend):
        self._test(*self.create_keras_gru_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_zero_recurrent_dropout = [
        dict(input_names=["x"], input_shapes=[[8, 2, 3]], input_type=tf.float32, units=3,
             activation='elu', recurrent_activation='tanh', dropouts=(.7, .0), use_bias=True,
             flags=(False, False)),
        dict(input_names=["x"], input_shapes=[[4, 8, 5]], input_type=tf.float32, dropouts=(.6, .0),
             units=2, activation='elu', use_bias=True, recurrent_activation='tanh',
             flags=(False, False)),
        dict(input_names=["x"], input_shapes=[[4, 3, 1]], input_type=tf.float32, units=8,
             activation='elu', recurrent_activation='tanh', dropouts=(.5, .0),
             use_bias=True, flags=(True, False)),
        dict(input_names=["x"], input_shapes=[[3, 4, 2]], input_type=tf.float32, units=3,
             activation='elu', recurrent_activation='tanh', dropouts=(.7, .0), use_bias=True,
             flags=(True, False)),
    ]

    @pytest.mark.parametrize("params", test_data_zero_recurrent_dropout)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="50176")
    def test_keras_gru_flags_zero_recurrent_dropout_float32(self, params, ie_device, precision,
                                                            temp_dir, ir_version,
                                                            use_legacy_frontend):
        self._test(*self.create_keras_gru_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
