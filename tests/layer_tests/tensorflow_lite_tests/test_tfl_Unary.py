# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import itertools
from functools import partial

import pytest
from common.tflite_layer_test_class import TFLiteLayerTest
import tensorflow as tf
import numpy as np

np.random.seed(42)


def make_positive_array(inputs_dict):
    for input in inputs_dict.keys():
        inputs_dict[input] = np.random.randint(1, 10, inputs_dict[input]).astype(np.float32)
    return inputs_dict


def make_boolean_array(inputs_dict):
    for input in inputs_dict.keys():
        inputs_dict[input] = np.random.randint(0, 1, inputs_dict[input]) > 1
    return inputs_dict


test_ops = [
    {'op_name': 'ABS', 'op_func': tf.math.abs},
    {'op_name': 'CAST', 'op_func': partial(tf.cast, dtype=tf.int32)},
    {'op_name': 'CEIL', 'op_func': tf.math.ceil},
    {'op_name': 'COS', 'op_func': tf.math.cos},
    {'op_name': 'ELU', 'op_func': tf.nn.elu},
    {'op_name': 'EXP', 'op_func': tf.math.exp, 'input_generator': make_positive_array},
    {'op_name': 'FLOOR', 'op_func': tf.math.floor},
    # {'op_name': 'HARD_SWISH'}, 'op_func': tf have no such operation
    {'op_name': 'LEAKY_RELU', 'op_func': partial(tf.nn.leaky_relu, alpha=-0.5)},
    {'op_name': 'LOG', 'op_func': tf.math.log, 'input_generator': make_positive_array},
    {'op_name': 'LOG_SOFTMAX', 'op_func': partial(tf.nn.log_softmax, axis=1)},
    {'op_name': 'LOGICAL_NOT', 'op_func': tf.math.logical_not, 'input_generator': make_boolean_array},
    {'op_name': 'LOGISTIC', 'op_func': tf.math.sigmoid},
    {'op_name': 'NEG', 'op_func': tf.math.negative},
    {'op_name': 'RANK', 'op_func': tf.rank},
    {'op_name': 'RELU6', 'op_func': tf.nn.relu6},
    {'op_name': 'ROUND', 'op_func': tf.math.round},
    {'op_name': 'RSQRT', 'op_func': tf.math.rsqrt, 'input_generator': make_positive_array},
    {'op_name': 'SHAPE', 'op_func': partial(tf.shape, out_type=tf.int32)},
    {'op_name': 'SIGN', 'op_func': tf.math.sign},
    {'op_name': 'SIN', 'op_func': tf.math.sin},
    {'op_name': 'SOFTMAX', 'op_func': partial(tf.nn.softmax, axis=1)},  # additionally test with alpha
    {'op_name': 'SQRT', 'op_func': tf.math.sqrt, 'input_generator': make_positive_array},
    {'op_name': 'SQUARE', 'op_func': tf.math.square},
    {'op_name': 'TANH', 'op_func': tf.math.tanh},
]

test_params = [
    {'shape': [2, 10, 10, 3]},
    {'shape': [2, 10]}
]
test_data = list(itertools.product(test_ops, test_params))


class TestTFLiteUnaryLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["UnaryOperation"]

    def _prepare_input(self, inputs_dict):
        if self.input_generator:
            return self.input_generator(inputs_dict)
        return super()._prepare_input(inputs_dict)

    @staticmethod
    def make_model(test_op_params: callable, shape: list):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            if test_op_params['op_name'] == 'LOGICAL_NOT':
                tf_input = tf.compat.v1.placeholder(tf.bool, shape, name=TestTFLiteUnaryLayerTest.inputs[0])
            else:
                tf_input = tf.compat.v1.placeholder(tf.float32, shape, name=TestTFLiteUnaryLayerTest.inputs[0])
            test_op_params['op_func'](tf_input, name=TestTFLiteUnaryLayerTest.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_unary(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
