import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import additional_test_params
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['L2_NORMALIZATION'], 'op_func': tf.math.l2_normalize},
]

test_params = [
    {'shape': [1]},
    {'shape': [1, 22]},
    {'shape': [1, 1, 8]},
    {'shape': [1, 22, 22, 8]},
]

test_data = parametrize_tests(test_ops, test_params)

test_data = list(itertools.product(test_data, additional_test_params[0]))
for i, (parameters, axis) in enumerate(test_data):
    parameters.update(axis)
    test_data[i] = parameters.copy()


class TestTFLiteExpandDimsLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["L2_Normalization"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            params['op_func'](place_holder, axis=params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_l2_normalization(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
