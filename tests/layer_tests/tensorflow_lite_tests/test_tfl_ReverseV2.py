import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': ['REVERSE_V2'], 'op_func': tf.reverse},
]

test_params = [
    {'shape': [1], 'axis': [-1]},
    {'shape': [1], 'axis': [0]},
    {'shape': [2, 6], 'axis': [-1, -2]},
    {'shape': [2, 6], 'axis': [1]},
    {'shape': [2, 4, 6], 'axis': [0, -2]},
    {'shape': [2, 4, 6], 'axis': [2]},
    {'shape': [2, 4, 6, 8], 'axis': [0, 3, -3, 2]},
    {'shape': [2, 4, 6, 8], 'axis': [-3]},
    {'shape': [2, 3, 1, 2, 2], 'axis': [0, 3, -3, 1, -1]},
    {'shape': [2, 3, 1, 2, 2], 'axis': [4]},
    {'shape': [2, 1, 1, 1, 2, 3, 2, 2], 'axis': [-1]},
    {'shape': [2, 1, 1, 1, 2, 3, 2, 2], 'axis': [0, 1, 2, 3, 4, 5, 6, 7]},
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteReverseV2LayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["ReverseV2"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'axis'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            params['op_func'](place_holder, params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_reverse_v2(self, params, ie_device, precision, temp_dir):
        if len(params['axis']) > 1:
            pytest.skip('CVS-109932')
        self._test(ie_device, precision, temp_dir, params)
