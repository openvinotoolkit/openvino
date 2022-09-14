# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasConvLSTM2D(CommonTF2LayerTest):
    def create_keras_conv_lstm_2d_net(self, params, input_names, input_shapes,
                                      input_type, ir_version):
        activation_func_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # tf.nn.<activation> operation have no "==" operation to be compared
            "relu": tf.nn.relu,
            "swish": tf.nn.swish,
            "elu": tf.nn.elu,
        }
        if "activation" in params:
            params["activation"] = activation_func_structure[params["activation"]]

        # create TensorFlow 2 model with Keras ConvLSTM2D operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state

        x = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.ConvLSTM2D(**params)(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [
        dict(params=dict(filters=16, kernel_size=(3, 3), padding='valid', return_sequences=False),
             input_names=["x"], input_shapes=[[10, 10, 300, 300, 1]], input_type=tf.float32),
        dict(params=dict(filters=16, kernel_size=(3, 3), padding='same', return_sequences=False,
                         activation="swish"),
             input_names=["x"], input_shapes=[[10, 10, 300, 300, 1]], input_type=tf.float32),
        dict(params=dict(filters=16, kernel_size=(3, 3), padding='valid', strides=(2, 2),
                         return_sequences=False),
             input_names=["x"], input_shapes=[[10, 10, 300, 300, 1]], input_type=tf.float32),
        dict(params=dict(filters=16, kernel_size=(3, 3), padding='valid', dilation_rate=3,
                         activation="relu", return_sequences=False, use_bias=True,
                         data_format="channels_last"),
             input_names=["x"], input_shapes=[[10, 10, 300, 300, 1]], input_type=tf.float32),
        dict(params=dict(filters=16, kernel_size=(3, 3), padding='valid', dilation_rate=3,
                         recurrent_activation="elu", return_sequences=False, use_bias=True,
                         data_format="channels_first"),
             input_names=["x"], input_shapes=[[10, 10, 1, 300, 300]], input_type=tf.float32)]

    # TODO: This test works only with tensorflow 2.3.0 or higher version
    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.xfail(reason="50141")
    def test_keras_conv_lstm_2d_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                        use_old_api, use_new_frontend):
        self._test(*self.create_keras_conv_lstm_2d_net(**params, ir_version=ir_version), ie_device,
                   precision,
                   temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
