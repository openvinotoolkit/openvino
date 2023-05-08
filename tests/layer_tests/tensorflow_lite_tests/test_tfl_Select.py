import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['SELECT'], 'op_func': tf.raw_ops.Select},
]

test_params = [
    {'shape': [2, 3, 1, 2, 2], 'condition': [True, False]},
    {'shape': [4, 3, 1, 2], 'condition': [False, False, False, True]},
    {'shape': [2, 3], 'condition': [[True, True, True], [False, False, False]]},
    {'shape': [2], 'condition': [False, True]},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteSelectLayerTest(TFLiteLayerTest):
    inputs = ["X", "Y"]
    outputs = ["Select"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'condition'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            cond = tf.constant(params['condition'], dtype=tf.bool)
            x = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            y = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[1])

            params['op_func'](condition=cond, x=x, y=y, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_select(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
