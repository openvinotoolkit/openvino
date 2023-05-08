import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': ['TOPK_V2'], 'op_func': tf.raw_ops.TopKV2},
]

test_params = [
    {'shape': [2], 'k': 2, 'sorted': True},
    {'shape': [2, 3], 'k': 1, 'sorted': False},
    {'shape': [2, 3, 5], 'k': 2, 'sorted': True},
    {'shape': [2, 3, 5, 10], 'k': 9, 'sorted': False},
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteTopKV2LayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["TopKV2"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'k', 'sorted'})) == 5, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            params['op_func'](input=placeholder, k=params['k'], sorted=params['sorted'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_topk_v2(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
