import numpy as np
import pytest
import tensorflow as tf
import random	
from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'shape': [random.randint(1, 30) for _ in range(4)], 'ksize': [random.randint(1, 10) for _ in range(4)], 'strides': [random.randint(1, 10) for _ in range(4)], 'padding': random.choice(['SAME',0]),
     'dilations': [random.randint(1, 10) for _ in range(2)]},
    {'shape': [random.randint(1, 24) for _ in range(4)], 'ksize': [random.randint(1, 10) for _ in range(4)], 'strides': [random.randint(1, 10) for _ in range(4)], 'padding': random.choice(['SAME',0]),
     'dilations': [random.randint(1, 10) for _ in range(2)]},
]


class TestTFLiteDepthwiseConv2DLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["DepthwiseConv2D"]
    allowed_ops = ['DEPTHWISE_CONV_2D']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'ksize', 'strides',
                                                    'padding', 'dilations'})) == 5, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            weights = tf.constant(np.random.randint(-1, 1, params['ksize']), dtype=tf.float32)
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.nn.depthwise_conv2d(place_holder, weights, params['strides'], params['padding'], 'NHWC',
                                   params['dilations'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_depthwise_conv2d(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
