import itertools

import pytest
import tensorflow as tf
import numpy as np

from common.tflite_layer_test_class import TFLiteLayerTest

test_ops = [
    {'op_name': ['RANGE'], 'op_func': tf.range},
]

test_params = [
    {'dtype': np.float32, 'negative_delta': False},
    {'dtype': np.int32, 'negative_delta': True},
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteRangeLayerTest(TFLiteLayerTest):
    inputs = ['Start', 'Limit', 'Delta']
    outputs = ["Range"]

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_data = {}
        if self.negative_delta:
            inputs_data['Start'] = np.random.randint(1, 10, []).astype(self.dtype)
            inputs_data['Limit'] = np.random.randint(-10, 0, []).astype(self.dtype)
            inputs_data['Delta'] = np.random.randint(-5, -1, []).astype(self.dtype)
        else:
            inputs_data['Start'] = np.random.randint(1, 10, []).astype(self.dtype)
            inputs_data['Limit'] = np.random.randint(10, 30, []).astype(self.dtype)
            inputs_data['Delta'] = np.random.randint(1, 5, []).astype(self.dtype)

        return inputs_data

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'dtype', 'negative_delta'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            self.dtype = params['dtype']
            self.negative_delta = params['negative_delta']

            start = tf.compat.v1.placeholder(self.dtype, [], self.inputs[0])
            limit = tf.compat.v1.placeholder(self.dtype, [], self.inputs[1])
            delta = tf.compat.v1.placeholder(self.dtype, [], self.inputs[2])

            params['op_func'](start, limit, delta, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_range(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
