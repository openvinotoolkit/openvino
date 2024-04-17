# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasGRUCell(CommonTF2LayerTest):
    def create_keras_grucell_net(self, input_names, input_shapes, input_type, units, activation,
                                 recurrent_activation,
                                 use_bias, reset_after, dropouts, ir_version):
        """
                create TensorFlow 2 model with Keras GRUCell operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        dropout, recurrent_dropout = dropouts
        cell = tf.keras.layers.GRUCell(units=units, activation=activation,
                                       recurrent_activation=recurrent_activation,
                                       use_bias=use_bias, reset_after=reset_after, dropout=dropout,
                                       recurrent_dropout=recurrent_dropout)
        y = tf.keras.layers.RNN(cell=cell, return_sequences=True, return_state=True)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=y)

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None

        return tf2_net, ref_net

    test_data_simple = [
        dict(input_names=["x"], input_shapes=[[2, 3, 4]], input_type=tf.float32, units=1,
             activation='relu', recurrent_activation='sigmoid', dropouts=(.0, .0), use_bias=True,
             reset_after=True),
        dict(input_names=["x"], input_shapes=[[5, 4, 6]], input_type=tf.float32, units=4,
             activation='elu', recurrent_activation='tanh', dropouts=(.2, .1), use_bias=False,
             reset_after=True),
        dict(input_names=["x"], input_shapes=[[3, 7, 5]], input_type=tf.float32, units=5,
             activation='linear', recurrent_activation='tanh', dropouts=(.9, .1), use_bias=True,
             reset_after=False),
        dict(input_names=["x"], input_shapes=[[5, 3, 3]], input_type=tf.float32, units=4,
             activation='softmax', recurrent_activation='sigmoid', dropouts=(.4, .2),
             use_bias=False, reset_after=False)
    ]

    @pytest.mark.parametrize("params", test_data_simple)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="49537")
    def test_keras_grucell_float32(self, params, ie_device, precision, temp_dir, ir_version,
                                   use_legacy_frontend):
        self._test(*self.create_keras_grucell_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
