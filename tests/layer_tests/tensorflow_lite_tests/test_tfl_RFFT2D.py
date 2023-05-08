import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['COMPLEX_ABS', 'RESHAPE', 'RFFT2D'], 'op_func': tf.signal.rfft2d},
]

test_params = [
    {'shape': [96, 1, 512], 'fft_length': [1, 512], 'reshape_to': [96, 257]},
    {'shape': [5, 1, 16], 'fft_length': [1, 16], 'reshape_to': [5, 9]},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteRFFT2DLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ['RFFT2D_1']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'fft_length', 'reshape_to'})) == 5, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                   name=self.inputs[0])
            rfft_2d = params['op_func'](placeholder, fft_length=params["fft_length"])
            reshape = tf.reshape(rfft_2d, params['reshape_to'])
            out = tf.raw_ops.ComplexAbs(x=reshape, Tout=tf.float32, name='RFFT2D')

            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_rfft2d(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
