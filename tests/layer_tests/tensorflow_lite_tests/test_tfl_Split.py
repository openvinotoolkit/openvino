import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [6], 'num_or_size_splits': 2, 'axis': 0},
    {'shape': [2, 1, 6], 'num_or_size_splits': 3, 'axis': 2},
    {'shape': [4, 3, 2, 7], 'num_or_size_splits': 4, 'axis': -4},
]


class TestTFLiteSplitLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Split"]
    allowed_ops = ['SPLIT']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'num_or_size_splits',
                                                    'axis'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            tf.split(placeholder, params["num_or_size_splits"], params["axis"], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_split(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
