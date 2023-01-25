# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from functools import partial

import pytest
from common.tflite_layer_test_class import TFLiteLayerTest
from common.layer_test_class import parametrize
import tensorflow as tf


test_ops = [
    ('ABS', tf.math.abs),
    ('CAST', partial(tf.cast, dtype=tf.int32)),
    ('CEIL', tf.math.ceil),
    ('COS', tf.math.cos),
    ('ELU', tf.nn.elu),
    ('EXP', tf.math.exp),
    ('FLOOR', tf.math.floor),
    # ('HARD_SWISH'), tf have no such operation
    ('LEAKY_RELU', partial(tf.nn.leaky_relu, alpha=-0.5)),
    ('LOG', tf.math.log),
    ('LOG_SOFTMAX', partial(tf.nn.log_softmax, axis=1)),
    ('LOGICAL_NOT', tf.math.logical_not),
    ('LOGISTIC', tf.math.sigmoid),
    ('NEG', tf.math.negative),
    ('RANK', tf.rank),
    ('RELU6', tf.nn.relu6),
    ('ROUND', tf.math.round),
    ('RSQRT', tf.math.rsqrt),
    ('SHAPE', partial(tf.shape, out_type=tf.int32)),
    ('SIGN', tf.math.sign),
    ('SIN', tf.math.sin),
    ('SOFTMAX', partial(tf.nn.softmax, axis=1)),  # additionally test with alpha
    ('SQRT', tf.math.sqrt),
    ('SQUARE', tf.math.square),
    ('TANH', tf.math.tanh),
    ('UNIQUE', tf.unique),
]

test_params = [
    {'shape': [2, 10, 10, 3]},
    {'shape': [2, 10]}
]
test_data = parametrize(test_ops, test_params)


class TestTFLiteUnaryLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["UnaryOperation"]

    @staticmethod
    def make_model(unary_op: callable, shape: list):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            tf_input = tf.compat.v1.placeholder(tf.float32, shape, name=TestTFLiteUnaryLayerTest.inputs[0])
            unary_op(tf_input, name=TestTFLiteUnaryLayerTest.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_unary(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
