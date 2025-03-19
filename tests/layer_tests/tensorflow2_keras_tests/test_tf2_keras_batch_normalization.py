# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasBatchNormalization(CommonTF2LayerTest):
    def create_keras_batch_normalization_net(self, axis, momentum, epsilon, center, scale,
                                             input_names, input_shapes,
                                             input_type, ir_version):
        # create TensorFlow 2 model with Keras BatchNormalization operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state

        inputs = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])

        y = tf.keras.layers.BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon,
                                               center=center, scale=scale)(inputs)
        tf2_net = tf.keras.Model(inputs=[inputs], outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [
        dict(axis=3, momentum=0.99, epsilon=1e-5, center=True, scale=True, input_names=["x1"],
             input_shapes=[[3, 4, 5, 6]], input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_batch_normalization_float32(self, params, ie_device, precision, ir_version,
                                               temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_batch_normalization_net(**params, ir_version=ir_version),
                   ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_extended_float32 = [dict(axis=1, momentum=0.5, epsilon=1e-4, center=True, scale=False,
                                       input_names=["x1"], input_shapes=[[3, 4]],
                                       input_type=tf.float32),
                                  dict(axis=1, momentum=0.3, epsilon=1e-3, center=False,
                                       scale=False,
                                       input_names=["x1"], input_shapes=[[3, 4, 5]],
                                       input_type=tf.float32),
                                  dict(axis=-1, momentum=0.0, epsilon=1e-5, center=True, scale=True,
                                       input_names=["x1"], input_shapes=[[3, 4, 5, 6]],
                                       input_type=tf.float32),
                                  dict(axis=2, momentum=0.99, epsilon=1e-2, center=False,
                                       scale=True,
                                       input_names=["x1"], input_shapes=[[3, 4, 5, 6, 7]],
                                       input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_extended_float32)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_keras_batch_normalization_extended_float32(self, params, ie_device, precision,
                                                        ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_batch_normalization_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
