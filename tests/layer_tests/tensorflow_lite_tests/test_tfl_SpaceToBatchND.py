import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': 'SPACE_TO_BATCH_ND', 'op_func': tf.space_to_batch},
]

test_params = [
    {'shape': [8, 10, 10, 3], 'block_shape': [2, 2], 'paddings': [[0, 2], [0, 0]]},
    {'shape': [24, 10, 10, 1], 'block_shape': [2, 12], 'paddings': [[2, 0], [0, 2]]}  # segfault
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteSpaceToBatchNDLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["SpaceToBatchOP"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'block_shape', 'paddings'})) == 5, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            params['op_func'](place_holder, params['block_shape'], params['paddings'],
                              name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_space_to_batch_nd(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
