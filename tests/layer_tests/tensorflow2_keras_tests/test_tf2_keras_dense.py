# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasDense(CommonTF2LayerTest):
    def create_keras_dense_net(self, input_names, input_shapes, input_type, units, activation,
                               use_bias, ir_version):
        """
            create TensorFlow 2 model with Keras Dense operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.Dense(units=units, activation=activation, use_bias=use_bias)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None

        return tf2_net, ref_net

    test_data_float32_simple = [
        dict(input_names=["x"], input_shapes=[[5, 4]], input_type=tf.float32, units=1,
             activation=None, use_bias=False),
        dict(input_names=["x"], input_shapes=[[5, 4, 8]], input_type=tf.float32, units=4,
             activation=None, use_bias=False),
        dict(input_names=["x"], input_shapes=[[5, 4, 8, 12]], input_type=tf.float32, units=16,
             activation=None, use_bias=False),
        dict(input_names=["x"], input_shapes=[[5, 4, 8, 6, 8]], input_type=tf.float32, units=10,
             activation=None, use_bias=True),
        dict(input_names=["x"], input_shapes=[[5, 4, 12, 3]], input_type=tf.float32, units=16,
             activation=None, use_bias=True),
        dict(input_names=["x"], input_shapes=[[5, 4, 2, 2, 2]], input_type=tf.float32, units=3,
             activation=None, use_bias=True)
    ]

    @pytest.mark.parametrize("params", test_data_float32_simple)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_dense_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_legacy_frontend):
        pytest.skip("Error: failed due to missing a required argument: x")
        self._test(*self.create_keras_dense_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_float32_activation = [
        dict(input_names=["x"], input_shapes=[[5, 4]], input_type=tf.float32, units=1,
             activation='relu', use_bias=False),
        dict(input_names=["x"], input_shapes=[[5, 4, 8]], input_type=tf.float32, units=4,
             activation='elu', use_bias=False),
        dict(input_names=["x"], input_shapes=[[5, 4]], input_type=tf.float32, units=1,
             activation='sigmoid', use_bias=True),
        pytest.param(dict(input_names=["x"], input_shapes=[[5, 4, 8]], input_type=tf.float32, units=4,
                          activation='tanh', use_bias=True), marks=pytest.mark.skip(reason="110006")),
        dict(input_names=["x"], input_shapes=[[5, 4, 8, 8]], input_type=tf.float32, units=5,
             activation='linear', use_bias=True),
        dict(input_names=["x"], input_shapes=[[5, 4, 8, 6, 4]], input_type=tf.float32, units=4,
             activation='softmax', use_bias=True)
    ]

    @pytest.mark.parametrize("params", test_data_float32_activation)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_activation_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        pytest.skip("Error: failed due to missing a required argument: x")
        self._test(*self.create_keras_dense_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
