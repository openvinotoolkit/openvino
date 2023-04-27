import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': 'BROADCAST_ARGS', 'op_func': tf.raw_ops.BroadcastArgs},  # tfl segfault
]

test_params = [
    {'shape': [5], 'broadcast_shape': [1], 'kwargs_to_prepare_input': 'int32'},
    {'shape': [1], 'broadcast_shape': [7], 'kwargs_to_prepare_input': 'int32'}
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteBroadcastToLayerTest(TFLiteLayerTest):
    inputs = ["Input", "Input1"]
    outputs = ["BroadcastArgsOP"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'broadcast_shape'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            s0 = tf.compat.v1.placeholder(params.get('dtype', tf.int32), params['shape'],
                                          name=self.inputs[0])
            s1 = tf.compat.v1.placeholder(params.get('dtype', tf.int32), params['broadcast_shape'],
                                          name=self.inputs[1])
            params['op_func'](s0=s0, s1=s1, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_broadcast_args(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
