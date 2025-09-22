# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasUpSampling1D(CommonTF2LayerTest):
    def create_keras_upsampling1d_net(self, input_names, input_shapes, input_type, size,
                                      ir_version):
        # create TensorFlow 2 model with Keras UpSampling1D operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0], dtype=input_type)
        y = tf.keras.layers.UpSampling1D(size=size)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    # Tests for float32 input
    test_data_float32 = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 8]], input_type=tf.float32, size=2),
        dict(input_names=["x1"], input_shapes=[[3, 2, 4]], input_type=tf.float32, size=3),
        dict(input_names=["x1"], input_shapes=[[1, 3, 8]], input_type=tf.float32, size=5),
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_upsampling1d_float32(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_keras_upsampling1d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)
