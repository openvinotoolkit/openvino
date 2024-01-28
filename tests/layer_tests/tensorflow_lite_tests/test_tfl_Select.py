import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 4) for _ in range(4)], 'condition': [True, False]},
    {'shape': [random.randint(1, 4) for _ in range(4)], 'condition': [False, False, False, True]},
    {'shape': [random.randint(1, 4) for _ in range(2)], 'condition': [[True, True, True], [False, False, False]]},
    {'shape': [random.randint(1, 4) for _ in range(1)], 'condition': [False, True]},
]


class TestTFLiteSelectLayerTest(TFLiteLayerTest):
    inputs = ["X", "Y"]
    outputs = ["Select"]
    allowed_ops = ['SELECT']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'condition'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            cond = tf.constant(params['condition'], dtype=tf.bool)
            x = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            y = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[1])

            tf.raw_ops.Select(condition=cond, x=x, y=y, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_select(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
