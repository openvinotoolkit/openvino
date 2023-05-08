import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': 'CONCATENATION', 'op_func': tf.concat},
]

test_params = [
    {'shapes': [[1, 3], [1, 7]], 'axis': -1},
    {'shapes': [[1, 3, 10], [1, 3, 56], [1, 3, 2]], 'axis': -1},
    {'shapes': [[1, 3, 10, 17], [1, 3, 10, 1], [1, 3, 10, 5], [1, 3, 10, 3]], 'axis': -1},
    {'shapes': [[2, 7], [2, 8]], 'axis': 1},
    {'shapes': [[1, 3, 10], [1, 2, 10], [1, 5, 10]], 'axis': 1},
    {'shapes': [[1, 3, 10, 17], [1, 10, 10, 17], [1, 12, 10, 17], [1, 1, 10, 17]], 'axis': 1},
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteConcatTest(TFLiteLayerTest):
    outputs = ['Concat']

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shapes', 'axis'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholders = []
            self.inputs = []
            for j, placeholder_shape in enumerate(params['shapes']):
                concat_input = f'concat_input_{j}'
                self.inputs.append(concat_input)
                placeholders.append(tf.compat.v1.placeholder(params.get('dtype', tf.float32), placeholder_shape,
                                                             name=concat_input))
            params['op_func'](placeholders, params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_concat(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
