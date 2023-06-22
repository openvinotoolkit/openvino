import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'shape': [1, 22, 22, 8], 'ksize': [32, 3, 4, 4], 'strides': 2, 'padding': 'SAME', 'dilations': [1, 1, 1, 1]},
    {'shape': [1, 22, 22, 9], 'ksize': [32, 3, 3, 3], 'strides': (2, 1), 'padding': 'SAME', 'dilations': [1, 2, 2, 1]},
    {'shape': [1, 22, 22, 8], 'ksize': [1, 3, 4, 4], 'strides': 2, 'padding': 'VALID', 'dilations': [1, 1, 1, 1]},
    {'shape': [1, 22, 22, 3], 'ksize': [1, 3, 3, 3], 'strides': (3, 4), 'padding': 'VALID', 'dilations': [1, 2, 2, 1]},
]


class TestTFLiteConv2DLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Conv2D"]
    allowed_ops = ['CONV_2D']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'ksize', 'strides',
                                                    'padding', 'dilations'})) == 5, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            weights = tf.constant(np.random.randint(-1, 1, params['ksize']), dtype=tf.float32)
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.nn.conv2d(place_holder, weights, params['strides'], params['padding'], 'NHWC',
                         params['dilations'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_conv2d(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
