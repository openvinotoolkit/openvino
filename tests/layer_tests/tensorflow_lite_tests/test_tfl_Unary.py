# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from functools import partial

import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

np.random.seed(42)

test_ops = [
    {'op_name': 'ABS', 'op_func': 'tf.math.abs'},
    {'op_name': 'CAST', 'op_func': 'partial(tf.cast, dtype=tf.int32)'},
    {'op_name': 'CEIL', 'op_func': 'tf.math.ceil'},
    {'op_name': 'COS', 'op_func': 'tf.math.cos'},
    {'op_name': 'ELU', 'op_func': 'tf.nn.elu'},
    {'op_name': 'EXP', 'op_func': 'tf.math.exp', 'kwargs_to_prepare_input': 'positive'},
    {'op_name': 'FLOOR', 'op_func': 'tf.math.floor'},
    {'op_name': 'LEAKY_RELU', 'op_func': 'partial(tf.nn.leaky_relu, alpha=-0.5)'},
    {'op_name': 'LOG', 'op_func': 'tf.math.log', 'kwargs_to_prepare_input': 'positive'},
    {'op_name': 'LOG_SOFTMAX', 'op_func': 'partial(tf.nn.log_softmax, axis=-1)'},
    {'op_name': 'LOGICAL_NOT', 'op_func': 'tf.math.logical_not', 'kwargs_to_prepare_input': 'boolean', 'dtype': tf.bool},
    {'op_name': 'LOGISTIC', 'op_func': 'tf.math.sigmoid'},
    {'op_name': 'NEG', 'op_func': 'tf.math.negative'},
    {'op_name': 'RELU6', 'op_func': 'tf.nn.relu6'},
    {'op_name': 'RELU', 'op_func': 'tf.nn.relu'},
    {'op_name': 'ROUND', 'op_func': 'tf.math.round'},
    {'op_name': 'RSQRT', 'op_func': 'tf.math.rsqrt', 'kwargs_to_prepare_input': 'positive'},
    {'op_name': 'SIN', 'op_func': 'tf.math.sin'},
    {'op_name': 'SOFTMAX', 'op_func': 'partial(tf.nn.softmax, axis=-1)'},  # additionally test with alpha
    {'op_name': 'SQRT', 'op_func': 'tf.math.sqrt', 'kwargs_to_prepare_input': 'positive'},
    {'op_name': 'SQUARE', 'op_func': 'tf.math.square'},
    {'op_name': 'TANH', 'op_func': 'tf.math.tanh'},

    # These operations are getting optimized out by tflite aka empty tfl model
    # {'op_name': 'RANK', 'op_func': tf.rank},
    # {'op_name': 'SHAPE', 'op_func': partial(tf.shape, out_type=tf.int32)},

    # This op could not be converted standalone -- tries to become FlexOp (offload from tfl to tf)
    # {'op_name': 'SIGN', 'op_func': tf.math.sign},
]

test_params = [
    {'shape': [2, 10, 10, 3]},
    {'shape': [2, 10]}
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteUnaryLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["UnaryOperation"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            eval(params['op_func'])(place_holder, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_unary(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
