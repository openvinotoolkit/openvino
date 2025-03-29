# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasAttention(CommonTF2LayerTest):
    def create_keras_attention_net(self, dropout, use_scale, input_names, input_shapes,
                                   input_type, ir_version):
        # create TensorFlow 2 model with Keras Attention operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        inputs = []
        for ind in range(len(input_names)):
            inputs.append(tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind]))
        y = tf.keras.layers.Attention(dropout=dropout, use_scale=use_scale)(inputs, training=False)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [
        dict(use_scale=True, dropout=0.5, input_names=["query", "value"],
             input_shapes=[[5, 4, 3], [5, 4, 3]], input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_attention_float32_case1(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        self._test(*self.create_keras_attention_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_extended_float32 = [
        dict(use_scale=False, dropout=0.0,
             input_names=["input1_query", "input2_value"],
             input_shapes=[[5, 4], [5, 4]], input_type=tf.float32),
        dict(use_scale=True, dropout=0.5,
             input_names=["input1_query", "input2_value"],
             input_shapes=[[2, 1, 4], [2, 1, 4]], input_type=tf.float32),
        dict(use_scale=False, dropout=0.8,
             input_names=["input1_query", "input2_value"],
             input_shapes=[[3, 2, 5], [3, 2, 5]], input_type=tf.float32),
        dict(use_scale=True, dropout=0.0,
             input_names=["input1_query", "input2_value", "input3_key"],
             input_shapes=[[4, 3], [4, 3], [4, 3]], input_type=tf.float32),
        dict(use_scale=False, dropout=0.5,
             input_names=["input1_query", "input2_value", "input3_key"],
             input_shapes=[[5, 3, 4], [5, 3, 4], [5, 3, 4]],
             input_type=tf.float32),
        dict(use_scale=True, dropout=0.5,
             input_names=["input1_query", "input2_value", "input3_key"],
             input_shapes=[[2, 1, 1], [2, 1, 1], [2, 1, 1]],
             input_type=tf.float32),
        dict(use_scale=False, dropout=0.8,
             input_names=["input1_query", "input2_value", "input3_key"],
             input_shapes=[[5, 3, 3], [5, 3, 3], [5, 3, 3]],
             input_type=tf.float32)
    ]

    # TODO: Extend test with const inputs, ticket: 49295
    @pytest.mark.parametrize("params", test_data_extended_float32)
    @pytest.mark.nightly
    def test_keras_attention_float32_case2(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        self._test(*self.create_keras_attention_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
