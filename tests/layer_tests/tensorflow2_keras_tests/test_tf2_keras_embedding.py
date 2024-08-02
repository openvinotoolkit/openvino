# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasEmbedding(CommonTF2LayerTest):
    def create_keras_emb_net(self, input_names, input_shapes, input_type, input_dim, output_dim,
                             mask_zero,
                             input_length, ir_version):
        """
                create TensorFlow 2 model with Keras Embedding operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0], dtype=input_type)
        layer = tf.keras.layers.PReLU()(x)
        y = tf.keras.layers.Embedding(input_dim, output_dim, mask_zero=mask_zero,
                                      input_length=input_length)(layer)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None
        return tf2_net, ref_net

    test_data_float32 = [
        dict(input_names=["x"], input_shapes=[[5, 16]], input_type=tf.float32, input_dim=256,
             output_dim=8, mask_zero=True, input_length=4),
        dict(input_names=["x"], input_shapes=[[5, 16]], input_type=tf.float32, input_dim=256,
             output_dim=324, mask_zero=True, input_length=16),
        dict(input_names=["x"], input_shapes=[[5, 16]], input_type=tf.float32, input_dim=256,
             output_dim=256, mask_zero=True, input_length=8),
        dict(input_names=["x"], input_shapes=[[5, 16]], input_type=tf.float32, input_dim=256,
             output_dim=32, mask_zero=True, input_length=4)
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_emb_float32(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(*self.create_keras_emb_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_mask_zero_false = [
        dict(input_names=["x"], input_shapes=[[5, 16]], input_type=tf.float32, input_dim=256,
             output_dim=32, mask_zero=False, input_length=1),
        dict(input_names=["x"], input_shapes=[[5, 16]], input_type=tf.float32, input_dim=256,
             output_dim=128, mask_zero=False, input_length=2),
        dict(input_names=["x"], input_shapes=[[5, 16]], input_type=tf.float32, input_dim=256,
             output_dim=256, mask_zero=False, input_length=4),
        dict(input_names=["x"], input_shapes=[[5, 16]], input_type=tf.float32, input_dim=256,
             output_dim=16, mask_zero=False, input_length=8)
    ]

    @pytest.mark.parametrize("params", test_data_mask_zero_false)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_emb_without_zero_mask_float32(self, params, ie_device, precision, ir_version,
                                                 temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_emb_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
