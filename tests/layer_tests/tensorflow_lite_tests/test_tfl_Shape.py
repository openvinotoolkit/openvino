# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from functools import partial

import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests


test_params = [
    {'shape': [1, 4], 'new_shape': [1, 4]},
    {'shape': [1, 4], 'new_shape': [4, 1]},
    {'shape': [1, 4], 'new_shape': [2, 2]},
]


class TestTFLiteShapeLayerTest(TFLiteLayerTest):
    inputs = ["Input", "Input1"]
    outputs = ["Shape"]
    allowed_ops = ['RESHAPE', 'SHAPE']

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_dict['Input'] = np.random.randint(0, 100, self.shape).astype(np.int32)
        inputs_dict['Input1'] = np.array(self.new_shape).astype(np.int32)
        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'new_shape'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        self.shape = params['shape']
        self.new_shape = params['new_shape']

        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.int32, self.shape, name=self.inputs[0])
            shape_of_new_shape = [len(self.new_shape)]
            new_shape = tf.compat.v1.placeholder(tf.int32, shape_of_new_shape, name=self.inputs[1])

            reshaped = tf.reshape(placeholder, shape=new_shape)
            tf.shape(input=reshaped)

            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_shape(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
