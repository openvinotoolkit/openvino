import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': 'ADD_N', 'op_func': tf.math.add_n},
]

test_params = [
    {'shape': [2, 10, 10, 3]},
    {'shape': [2, 10]}
]

test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteAddNLayerTest(TFLiteLayerTest):
    inputs = ["Input_0", "Input_1"]
    outputs = ["AddN"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder0 = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                     name=self.inputs[0]),
            place_holder1 = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                     name=self.inputs[1])
            params['op_func']([place_holder0[0], place_holder1], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_add_n(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
