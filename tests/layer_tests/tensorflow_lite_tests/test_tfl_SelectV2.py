import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 4) for _ in range(5)], 'condition': [True, False]},
    {'shape': [random.randint(1, 4) for _ in range(4)], 'condition': [False, True]},
    {'shape': [random.randint(1, 4) for _ in range(4)], 'condition': [True, True, False]},
    {'shape': [random.randint(1, 4) for _ in range(3)], 'condition': [True, True, False]},
    {'shape': [random.randint(1, 4) for _ in range(2)], 'condition': [True, False, True]},
]


class TestTFLiteSelectLayerTest(TFLiteLayerTest):
    inputs = ["X", "Y"]
    outputs = ["Select"]
    allowed_ops = ['SELECT_V2']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'condition'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            cond = tf.constant(params['condition'], dtype=tf.bool)
            x = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            y = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[1])

            tf.raw_ops.SelectV2(condition=cond, t=x, e=y, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_select_v2(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
