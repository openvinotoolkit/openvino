# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasConv1DTranspose(CommonTF2LayerTest):
    def create_keras_conv1d_transpose_net(self, params, input_names, input_shapes, input_type,
                                          ir_version):
        activation_func_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # tf.nn.<activation> operation have no "==" operation to be compared
            "relu": tf.nn.relu
        }
        params = params.copy()
        if "activation" in params:
            params["activation"] = activation_func_structure[params["activation"]]

        # create TensorFlow 2 model with Keras Conv1DTranspose operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], dtype=input_type,
                           name=input_names[0])  # Variable-length sequence of ints

        y = tf.keras.layers.Conv1DTranspose(**params, input_shape=input_shapes[0][1:])(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32_set1 = [
        dict(params=dict(filters=27, kernel_size=3, padding="valid", strides=2),
             input_names=["x"],
             input_shapes=[[5, 7, 6]], input_type=tf.float32),
        dict(params=dict(filters=10, kernel_size=5, padding="same", activation="relu",
                         use_bias=True),
             input_names=["x"], input_shapes=[[5, 7, 8]], input_type=tf.float32),
        dict(params=dict(filters=27, kernel_size=3, padding="valid", dilation_rate=3),
             input_names=["x"],
             input_shapes=[[5, 7, 6]], input_type=tf.float32),
        dict(
            params=dict(filters=20, kernel_size=7, padding="valid", data_format="channels_first"),
            input_names=["x"], input_shapes=[[5, 7, 8]], input_type=tf.float32),
        dict(params=dict(filters=10, kernel_size=5, padding="same", strides=3), input_names=["x"],
             input_shapes=[[5, 7, 8]], input_type=tf.float32),
        dict(params=dict(filters=20, kernel_size=7, padding="valid", strides=4), input_names=["x"],
             input_shapes=[[5, 7, 8]], input_type=tf.float32),
        dict(params=dict(filters=27, kernel_size=3, padding="valid", dilation_rate=3),
             input_names=["x"],
             input_shapes=[[5, 7, 6]], input_type=tf.float32),
        dict(params=dict(filters=20, kernel_size=7, padding="valid", data_format="channels_first"),
             input_names=["x"],
             input_shapes=[[5, 7, 8]], input_type=tf.float32),
    ]

    # TODO: This test works only with tensorflow 2.3.0 or higher version
    @pytest.mark.parametrize("params", test_data_float32_set1)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(reason="Needs tensorflow 2.3.0.")
    def test_keras_conv_1d_case1_transpose_float32(self, params, ie_device, precision, ir_version,
                                                   temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_conv1d_transpose_net(**params, ir_version=ir_version),
                   ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
