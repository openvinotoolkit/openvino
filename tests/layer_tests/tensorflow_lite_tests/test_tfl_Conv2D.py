import itertools

import pytest
import tensorflow as tf
import numpy as np

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': ['CONV_2D'], 'op_func': tf.nn.conv2d},
]

test_params = [
    {'shape': [1, 22, 22, 8], 'ksize': [32, 3, 4, 4], 'strides': 2, 'padding': 'SAME', 'data_format': 'NHWC',
     'dilations': [1, 1, 1, 1]},
    {'shape': [1, 22, 22, 9], 'ksize': [32, 3, 3, 3], 'strides': (2, 1), 'padding': 'SAME', 'data_format': 'NHWC',
     'dilations': [1, 2, 2, 1]},
    {'shape': [1, 22, 22, 8], 'ksize': [1, 3, 4, 4], 'strides': 2, 'padding': 'VALID', 'data_format': 'NHWC',
     'dilations': [1, 1, 1, 1]},
    {'shape': [1, 22, 22, 3], 'ksize': [1, 3, 3, 3], 'strides': (3, 4), 'padding': 'VALID', 'data_format': 'NHWC',
     'dilations': [1, 2, 2, 1]},
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteConv2DLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Conv2D"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'ksize', 'strides',
                                                    'padding', 'data_format', 'dilations'})) == 8, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            weights = tf.constant(np.random.randint(-1, 1, params['ksize']), dtype=tf.float32)
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            params['op_func'](place_holder, weights, params['strides'], params['padding'], params['data_format'],
                              params['dilations'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_conv2d(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
