import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': 'GATHER_ND', 'op_func': tf.gather_nd},
]

test_params = [
    {'shape': [1, 10], 'indices_shape': [1, 2]},
    {'shape': [2, 3, 5, 5], 'indices_shape': [4, 4]},
    {'shape': [1, 5, 5], 'indices_shape': [2, 4, 3]},
    {'shape': [1, 5, 5], 'indices_shape': [2, 4, 5, 3]},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteGatherLayerTest(TFLiteLayerTest):
    inputs = ["Input_x"]
    outputs = ["GatherND"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'indices_shape', })) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32),
                                                   params['shape'], name=self.inputs[0])
            constant = tf.constant(np.random.randint(0, 1, size=params['indices_shape']))

            params['op_func'](placeholder, constant, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_gather_nd(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
