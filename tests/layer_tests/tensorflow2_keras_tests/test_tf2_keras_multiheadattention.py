# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasMultiHeadAttention(CommonTF2LayerTest):
    def create_keras_multiheadattention_net(self,
                                            input_names, input_shapes, input_types,
                                            attention_mask_value,
                                            num_heads, key_dim, value_dim, dropout, use_bias,
                                            output_shape, attention_axes,
                                            return_attention_scores, training,
                                            ir_version):
        # create TensorFlow 2 model with Keras MultiHeadAttention operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state

        query = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0], dtype=input_types[0])
        value = tf.keras.Input(shape=input_shapes[1][1:], name=input_names[1], dtype=input_types[1])
        inputs = [query, value]
        key = None
        if len(input_shapes) > 2:
            key = tf.keras.Input(shape=input_shapes[2][1:], name=input_names[2],
                                 dtype=input_types[2])
            inputs.append(key)
        attention_mask = None
        if attention_mask_value is not None:
            attention_mask = tf.keras.Input(tensor=attention_mask_value)
            inputs.append(attention_mask)

        y = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                               value_dim=value_dim,
                                               dropout=dropout, use_bias=use_bias,
                                               output_shape=output_shape,
                                               attention_axes=attention_axes)(query=query,
                                                                              value=value,
                                                                              key=key,
                                                                              attention_mask=attention_axes,
                                                                              return_attention_scores=return_attention_scores,
                                                                              training=training)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    # Now all test cases fail due to Einsum operation is not supported
    # Tests with default attributes
    test_data = [
        pytest.param(
            dict(input_names=["query", "value"], input_shapes=[[2, 8, 16], [2, 4, 16]],
                 input_types=[tf.float32, tf.float32], attention_mask_value=None,
                 num_heads=2, key_dim=3, value_dim=4, dropout=0.0, use_bias=True,
                 output_shape=None, attention_axes=None,
                 return_attention_scores=False, training=False),
            marks=pytest.mark.xfail(reason="45432")),
        pytest.param(
            dict(input_names=["query", "value", "key"],
                 input_shapes=[[3, 8, 16], [3, 5, 16], [3, 5, 16]],
                 input_types=[tf.float32, tf.float32, tf.float32], attention_mask_value=None,
                 num_heads=3, key_dim=4, value_dim=2, dropout=0.0, use_bias=True,
                 output_shape=None, attention_axes=None,
                 return_attention_scores=False, training=False),
            marks=pytest.mark.xfail(reason="45432")),
        pytest.param(
            dict(input_names=["query", "value", "key"],
                 input_shapes=[[1, 8, 16], [1, 2, 4], [1, 2, 4]],
                 input_types=[tf.float32, tf.float32, tf.float32], attention_mask_value=None,
                 num_heads=1, key_dim=3, value_dim=4, dropout=0.0, use_bias=True,
                 output_shape=None, attention_axes=None,
                 return_attention_scores=True, training=False),
            marks=pytest.mark.xfail(reason="45432"))
    ]

    @pytest.mark.skip(reason='Einsum is unsupported in OVC')
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit
    def test_keras_multiheadattention(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_keras_multiheadattention_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    # Tests to cover no bias cases
    test_data_no_bias = [
        pytest.param(
            dict(input_names=["query", "value"], input_shapes=[[3, 5, 7], [3, 4, 16]],
                 input_types=[tf.float32, tf.float32], attention_mask_value=None,
                 num_heads=2, key_dim=3, value_dim=4, dropout=0.0, use_bias=True,
                 output_shape=None, attention_axes=None,
                 return_attention_scores=False, training=False),
            marks=pytest.mark.xfail(reason="45432")),
        pytest.param(
            dict(input_names=["query", "value", "key"],
                 input_shapes=[[2, 8, 16], [2, 4, 16], [2, 4, 16]],
                 input_types=[tf.float32, tf.float32, tf.float32], attention_mask_value=None,
                 num_heads=3, key_dim=4, value_dim=2, dropout=0.0, use_bias=False,
                 output_shape=None, attention_axes=None,
                 return_attention_scores=False, training=False),
            marks=pytest.mark.xfail(reason="45432"))
    ]

    @pytest.mark.skip(reason='Einsum is unsupported in OVC')
    @pytest.mark.parametrize("params", test_data_no_bias)
    @pytest.mark.nightly
    def test_keras_multiheadattention_no_bias(self, params, ie_device, precision, ir_version,
                                              temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_multiheadattention_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
