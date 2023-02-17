import itertools

import tensorflow as tf
import pytest

from common.tflite_layer_test_class import TFLiteLayerTest
from tensorflow_lite_tests.test_tfl_Unary import data_generators

test_ops = [
    {'op_name': 'ADD', 'op_func': tf.math.add},
    {'op_name': 'DIV', 'op_func': tf.math.divide, 'kwargs_to_prepare_input': 'positive'},
    {'op_name': 'MUL', 'op_func': tf.math.multiply},
    {'op_name': 'SUB', 'op_func': tf.math.subtract},
]

test_params = [
    {'shape': [2, 10, 10, 3]},
    {'shape': [2, 10]}
]

activations = [
    {'activation': None},
    {'activation': tf.nn.relu},
    {'activation': tf.nn.relu6},
    {'activation': tf.math.tanh},
    {'activation': tf.experimental.numpy.signbit},
    # n1
]

test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()

test_data = list(itertools.product(test_data, activations))
for i, (parameters, activations) in enumerate(test_data):
    parameters.update(activations)
    test_data[i] = parameters.copy()


def activation_helper(input_node, activation_name: callable):
    if activation_name is None:
        return input_node
    else:
        return activation_name(input_node)


class TestTFLiteBinaryWith_ActivationLayerTest(TFLiteLayerTest):
    inputs = ["Input_0", "Input_1"]
    outputs = ["BinaryOperation"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'activation'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            in0 = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                           name=TestTFLiteBinaryWith_ActivationLayerTest.inputs[0])
            in1 = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                           name=TestTFLiteBinaryWith_ActivationLayerTest.inputs[1])
            op = params['op_func'](in0, in1, name=TestTFLiteBinaryWith_ActivationLayerTest.outputs[0])
            op = activation_helper(op, params['activation'])

            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_binary(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
