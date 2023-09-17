import numpy as np
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
