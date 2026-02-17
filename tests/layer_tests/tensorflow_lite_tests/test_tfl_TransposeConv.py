# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'shape': [1, 3, 4, 1], 'ksize': [1, 1, 1, 1], 'output_shape': [1, 3, 4, 1], 'strides': [1, 1, 1, 1],
     'padding': 'SAME', 'dilations': [1, 1, 1, 1]},
    {'shape': [1, 4, 4, 1], 'ksize': [1, 1, 1, 1], 'output_shape': [1, 4, 4, 1], 'strides': [1, 1], 'padding': 'SAME',
     'dilations': [1, 2, 2, 1]},
    {'shape': [1, 22, 22, 3], 'ksize': [1, 1, 6, 3], 'output_shape': [1, 22, 22, 6], 'strides': [1, 1],
     'padding': 'VALID', 'dilations': [1, 1, 1, 1]},
    {'shape': [1, 22, 22, 3], 'ksize': [3, 3, 3, 3], 'output_shape': [1, 24, 24, 3], 'strides': [1, 1],
     'padding': 'VALID', 'dilations': [1, 1, 1, 1]},
]


class TestTFLiteTransposeConvLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["TransposeConv"]
    allowed_ops = ['TRANSPOSE_CONV']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'ksize', 'strides',
                                                    'padding', 'dilations', 'output_shape'})) == 6, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                   name=self.inputs[0])
            filter_input = tf.constant(np.random.randint(-1, 1, size=(params['ksize'])), dtype=tf.float32)
            tf.nn.conv2d_transpose(placeholder, filter_input, params['output_shape'], params["strides"],
                                   params["padding"], 'NHWC', name=self.outputs[0])

            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_transpose_conv(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)


test_params_with_bias = [
    {'shape': [1, 3, 4, 1], 'filters': 1, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same'},
    {'shape': [1, 4, 4, 1], 'filters': 1, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same'},
    {'shape': [1, 22, 22, 3], 'filters': 6, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid'},
    {'shape': [1, 22, 22, 3], 'filters': 3, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'valid'},
]


class TestTFLiteTransposeConvWithBiasLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["TransposeConvBias"]
    allowed_ops = ['TRANSPOSE_CONV']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'filters', 'kernel_size',
                                                    'strides', 'padding'})) == 5, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        # Use Keras Conv2DTranspose with use_bias=True so the TFLite converter
        # produces a single TRANSPOSE_CONV op with 4 inputs (including bias).
        input_shape = params['shape'][1:]  # exclude batch dim
        inp = tf.keras.Input(shape=input_shape, batch_size=params['shape'][0], name=self.inputs[0])
        # Fix random seeds for reproducibility of kernel and bias initializers
        out = tf.keras.layers.Conv2DTranspose(
            filters=params['filters'],
            kernel_size=params['kernel_size'],
            strides=params['strides'],
            padding=params['padding'],
            use_bias=True,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=42),
            name=self.outputs[0],
        )(inp)
        self.keras_model = tf.keras.Model(inputs=inp, outputs=out)
        # Return None — we override produce_tflite_model to use the Keras model directly
        return None

    def produce_tflite_model(self, framework_model, save_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)
        tflite_model = converter.convert()

        tflite_model_path = os.path.join(save_path, 'model.tflite')
        with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        return tflite_model_path

    @pytest.mark.parametrize("params", test_params_with_bias)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_transpose_conv_with_bias(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
