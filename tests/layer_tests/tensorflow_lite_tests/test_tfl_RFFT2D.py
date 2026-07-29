import platform

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [96, 1, 512], 'fft_length': [1, 512], 'reshape_to': [96, 257]},
    {'shape': [5, 1, 16], 'fft_length': [1, 16], 'reshape_to': [5, 9]},
]


class TestTFLiteRFFT2DLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ['RFFT2D_1']
    allowed_ops = ['COMPLEX_ABS', 'RESHAPE', 'RFFT2D']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'fft_length', 'reshape_to'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                   name=self.inputs[0])
            rfft_2d = tf.signal.rfft2d(placeholder, fft_length=params["fft_length"])
            reshape = tf.reshape(rfft_2d, params['reshape_to'])
            out = tf.raw_ops.ComplexAbs(x=reshape, Tout=tf.float32, name='RFFT2D')

            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    @pytest.mark.xfail(platform.machine() in ["aarch64", "arm64", "ARM64"],
                       reason='Ticket - 123324')
    def test_rfft2d(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
