import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': 'FULLY_CONNECTED', 'op_func': tf.matmul}
]

test_params = [
    {'shape_x': [40, 37], 'shape_y': [37, 37], 'transpose_a': False, 'transpose_b': True},
    {'shape_x': [5, 5], 'shape_y': [4, 5], 'transpose_a': False, 'transpose_b': True},
    {'shape_x': [1, 5, 5], 'shape_y': [4, 5], 'transpose_a': False, 'transpose_b': True},
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteFUllyConnectedLayerTest(TFLiteLayerTest):
    inputs = ["Input_x", "Input_y"]
    outputs = ["FullyConnected"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape_x', 'shape_y'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape_x'], name=self.inputs[0])
            y = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape_y'], name=self.inputs[1])

            params['op_func'](x, y, transpose_a=params['transpose_a'], transpose_b=params['transpose_b'],
                              name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_fully_connected(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
