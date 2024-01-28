import numpy as np
import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'shape': [random.randint(1, 10) for _ in range(4)], 'ksize': [random.randint(1, 10) for _ in range(4)], 'output_shape': [random.randint(1, 10) for _ in range(4)], 'strides': [random.randint(1, 10) for _ in range(4)],
     'padding': 'SAME', 'dilations': [random.randint(1, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(4)], 'ksize': [random.randint(1, 10) for _ in range(4)], 'output_shape': [random.randint(1, 10) for _ in range(4)], 'strides': [random.randint(1, 10) for _ in range(2)], 'padding': 'SAME',
     'dilations': [random.randint(1, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 20) for _ in range(4)], 'ksize': [random.randint(1, 10) for _ in range(4)], 'output_shape': [random.randint(1, 24) for _ in range(4)], 'strides': [random.randint(1, 10) for _ in range(2)],
     'padding': 'VALID', 'dilations': [random.randint(1, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 24) for _ in range(4)], 'ksize': [random.randint(1, 10) for _ in range(4)], 'output_shape': [random.randint(1, 34) for _ in range(4)], 'strides': [random.randint(1, 10) for _ in range(2)],
     'padding': 'VALID', 'dilations': [random.randint(1, 10) for _ in range(4)]},
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
