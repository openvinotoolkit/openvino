# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasLayerNormalization(CommonTF2LayerTest):
    def create_keras_lnorm_net(self, input_names, input_shapes, input_type, axis, epsilon, center,
                               scale, ir_version):
        """
               create TensorFlow 2 model with Keras LayerNormalization operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.LayerNormalization(axis=axis, epsilon=epsilon, center=center,
                                               scale=scale,
                                               trainable=False)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None

        return tf2_net, ref_net

    test_data_float32_precommit = [
        dict(input_names=["x"], input_shapes=[[2, 2, 3, 5]], input_type=tf.float32,
             axis=(1, 2, 3), epsilon=1e-5, center=True, scale=True)]

    @pytest.mark.parametrize("params", test_data_float32_precommit)
    @pytest.mark.precommit
    def test_keras_dense_float32(self, params, ie_device, precision, temp_dir, ir_version,
                                 use_legacy_frontend):
        self._test(*self.create_keras_lnorm_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_float32 = [
        dict(input_names=["x"], input_shapes=[[5, 10]], input_type=tf.float32, axis=1, epsilon=1e-6,
             center=False, scale=False),
        dict(input_names=["x"], input_shapes=[[3, 3, 5]], input_type=tf.float32, axis=1,
             epsilon=1e-7,
             center=True, scale=False),
        dict(input_names=["x"], input_shapes=[[2, 3, 8]], input_type=tf.float32, axis=2,
             epsilon=1e-6,
             center=False, scale=True),
        dict(input_names=["x"], input_shapes=[[2, 2, 3, 5]], input_type=tf.float32, axis=(1, 2, 3),
             epsilon=1e-5,
             center=True, scale=True)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_keras_dense_float32(self, params, ie_device, precision, temp_dir, ir_version):
        self._test(*self.create_keras_lnorm_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)
