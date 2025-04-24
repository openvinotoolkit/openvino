import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [2], 'multiples': [2]},
    {'shape': [2, 3], 'multiples': [1, 2]},
    {'shape': [2, 3, 1], 'multiples': [5, 1, 3]},
]


class TestTFLiteTileLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Tile"]
    allowed_ops = ['TILE']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'multiples'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            tf.tile(placeholder, params['multiples'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_tile(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
