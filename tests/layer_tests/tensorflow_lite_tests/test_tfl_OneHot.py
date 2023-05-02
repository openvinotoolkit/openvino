import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': ['ONE_HOT'], 'op_func': tf.one_hot},
]

test_params = [
    {'shape': [3], 'axis': 0, 'kwargs_to_prepare_input': 'int32_positive'},
    {'shape': [4, 4], 'axis': 1},
    {'shape': [1, 5, 3], 'axis': 0},
    {'shape': [5, 1, 2, 4], 'axis': 1},
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteOneHotLayerTest(TFLiteLayerTest):
    inputs = ['Indices', 'Depth', 'OnValue', 'OffValue']
    outputs = ["OneHot"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'axis'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            indices = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[0], shape=params["shape"])
            depth = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[1], shape=())

            on_value = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[2], shape=())
            off_value = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[3], shape=())

            params['op_func'](indices=indices, depth=depth, on_value=on_value, off_value=off_value,
                              axis=params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_one_hot(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
