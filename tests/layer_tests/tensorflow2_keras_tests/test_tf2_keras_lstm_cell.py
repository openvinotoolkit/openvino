# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasLSTMCell(CommonTF2LayerTest):
    def create_keras_lstmcell_net(self, input_names, input_shapes, input_type, units, activation,
                                  recurrent_activation,
                                  use_bias, unit_forget_bias, dropouts, ir_version):
        """
                create TensorFlow 2 model with Keras LSTMCell operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        dropout, recurrent_dropout = dropouts
        cell = tf.keras.layers.LSTMCell(units=units, activation=activation,
                                        recurrent_activation=recurrent_activation,
                                        use_bias=use_bias, unit_forget_bias=unit_forget_bias,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout)
        y = tf.keras.layers.RNN(cell=cell, return_sequences=True, return_state=True)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=y)

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None

        return tf2_net, ref_net

    test_data_simple = [
        dict(input_names=["x"], input_shapes=[[5, 5, 4]], input_type=tf.float32, units=1,
             activation='softmax',
             recurrent_activation='sigmoid', dropouts=(.0, .0), use_bias=True,
             unit_forget_bias=True),
        dict(input_names=["x"], input_shapes=[[2, 8, 2]], input_type=tf.float32, units=4,
             activation='relu',
             recurrent_activation='sigmoid', dropouts=(.2, .1), use_bias=False,
             unit_forget_bias=True),
        dict(input_names=["x"], input_shapes=[[3, 4, 3]], input_type=tf.float32, units=5,
             activation='softsign',
             recurrent_activation='elu', dropouts=(.9, .1), use_bias=True, unit_forget_bias=False),
        dict(input_names=["x"], input_shapes=[[1, 6, 5]], input_type=tf.float32, units=6,
             activation='softplus',
             recurrent_activation='softsign', dropouts=(.4, .2), use_bias=False,
             unit_forget_bias=False)
    ]

    @pytest.mark.parametrize("params", test_data_simple)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="49537")
    def test_keras_lstmcell_float32(self, params, ie_device, precision, temp_dir, ir_version,
                                    use_old_api, use_new_frontend):
        self._test(*self.create_keras_lstmcell_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
