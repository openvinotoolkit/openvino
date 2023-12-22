import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [2, 3], 'value': 1, 'kwargs_to_prepare_input': 'int32_positive'},
    {'shape': [2, 3, 3, 4], 'value': -1, 'kwargs_to_prepare_input': 'int32_positive'},
    {'shape': [1], 'value': 0, 'kwargs_to_prepare_input': 'int32_positive'}
]


class TestTFLiteBroadcastToLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Fill"]
    allowed_ops = ['FILL']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'value'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.int32), [len(params['shape'])],
                                                   name=self.inputs[0])
            tf.fill(placeholder, params['value'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_fill(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
