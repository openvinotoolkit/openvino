import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 10) for _ in range(1)], 'axis': random.randint(0,1), 'kwargs_to_prepare_input': 'int32_positive'},
    {'shape': [random.randint(1, 10) for _ in range(2)], 'axis': random.randint(0,1), 'kwargs_to_prepare_input': 'int32_positive'},
    {'shape': [random.randint(1, 10) for _ in range(3)], 'axis': random.randint(0,1), 'kwargs_to_prepare_input': 'int32_positive'},
    {'shape': [random.randint(1, 10) for _ in range(4)], 'axis': random.randint(0,1), 'kwargs_to_prepare_input': 'int32_positive'},
]


class TestTFLiteOneHotLayerTest(TFLiteLayerTest):
    inputs = ['Indices', 'Depth', 'OnValue', 'OffValue']
    outputs = ["OneHot"]
    allowed_ops = ['ONE_HOT']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'axis'})) == 1, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            indices = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[0], shape=params["shape"])
            depth = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[1], shape=())

            on_value = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[2], shape=())
            off_value = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[3], shape=())

            tf.one_hot(indices=indices, depth=depth, on_value=on_value, off_value=off_value,
                       axis=params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_one_hot(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
