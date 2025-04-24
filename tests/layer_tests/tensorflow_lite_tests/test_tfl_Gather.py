import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'shape': [1, 10], 'indices_shape': [1, 5]},
    {'shape': [2, 3, 5, 5], 'indices_shape': [4, 5]},
    {'shape': [1, 5, 5], 'indices_shape': [2, 4, 5]},
    {'shape': [1, 5, 5], 'indices_shape': [2, 4, 5, 6]},
]


class TestTFLiteGatherLayerTest(TFLiteLayerTest):
    inputs = ["Input_x"]
    outputs = ["Gather"]
    allowed_ops = ['GATHER']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'indices_shape', })) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32),
                                                   params['shape'], name=self.inputs[0])
            constant = tf.constant(np.random.randint(0, 1, size=params['indices_shape']))

            tf.gather(placeholder, constant, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_gather(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
