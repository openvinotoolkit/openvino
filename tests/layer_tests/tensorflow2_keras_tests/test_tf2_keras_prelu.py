# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasPReLU(CommonTF2LayerTest):
    def create_keras_prelu_net(self, input_names, input_shapes, input_type, shared_axes,
                               ir_version):
        # create TensorFlow 2 model with Keras PReLU operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.PReLU(shared_axes=shared_axes)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32_precommit = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]], input_type=tf.float32,
             shared_axes=None)]

    @pytest.mark.parametrize("params", test_data_float32_precommit)
    @pytest.mark.precommit
    def test_keras_prelu_float32(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_keras_prelu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)

    test_data_float32 = [
        dict(input_names=["x1"], input_shapes=[[5, 4]], input_type=tf.float32, shared_axes=None),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8]], input_type=tf.float32, shared_axes=None),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3]], input_type=tf.float32,
             shared_axes=None),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]], input_type=tf.float32,
             shared_axes=None)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_keras_prelu_float32(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_keras_prelu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)

    test_data_float32_shared_axes = [
        dict(input_names=["x1"], input_shapes=[[5, 4]], input_type=tf.float32, shared_axes=[1]),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8]], input_type=tf.float32, shared_axes=[1]),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3]], input_type=tf.float32,
             shared_axes=[1, 2]),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]], input_type=tf.float32,
             shared_axes=[1, 2, 3])]

    @pytest.mark.parametrize("params", test_data_float32_shared_axes)
    @pytest.mark.nightly
    def test_keras_prelu_float32_shared_axes(self, params, ie_device, precision, ir_version,
                                             temp_dir):
        self._test(*self.create_keras_prelu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)
