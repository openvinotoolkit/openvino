# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasBidirectional(CommonTF2LayerTest):
    def create_keras_bidirectional_net(self, n_units, RNN_layer, input_names, input_shapes,
                                       input_type, ir_version):
        RNN_layer_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # tf.keras.layers.<cell> operation have no "==" operation to be compared
            "LSTM": tf.keras.layers.LSTM,
            "GRU": tf.keras.layers.GRU,
            "SimpleRNN": tf.keras.layers.SimpleRNN
        }
        RNN_layer = RNN_layer_structure[RNN_layer]

        # create TensorFlow 2 model with Keras Bidirectional operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state

        x = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        forward_layer = RNN_layer(n_units, return_sequences=False)
        backward_layer = RNN_layer(n_units, activation='relu', return_sequences=False,
                                   go_backwards=True)
        bd_layer = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                                                 input_shape=input_shapes[0][1:])
        tf2_net = tf.keras.Model(inputs=[x], outputs=bd_layer(x))

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [
        dict(n_units=3, RNN_layer="LSTM", input_names=["x"], input_shapes=[[3, 4, 5]],
             input_type=tf.float32),
        dict(n_units=4, RNN_layer="GRU", input_names=["x"], input_shapes=[[3, 4, 5]],
             input_type=tf.float32),
        dict(n_units=5, RNN_layer="SimpleRNN", input_names=["x"],
             input_shapes=[[3, 4, 5]], input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_keras_bidirectional_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                         use_old_api, use_new_frontend):
        self._test(*self.create_keras_bidirectional_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
