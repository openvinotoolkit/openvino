import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['MIRROR_PAD'], 'op_func': tf.pad},
]

test_params = [
    {'shape': [2, 3], 'padding_matrix': [[1, 1], [2, 1]], 'mode': 'REFLECT'},
    {'shape': [2, 3], 'padding_matrix': [[1, 1], [1, 1]], 'mode': 'REFLECT'},
    {'shape': [2, 3], 'padding_matrix': [[1, 1], [2, 1]], 'mode': 'SYMMETRIC'},

    {'shape': [3], 'padding_matrix': [[0, 2]], 'mode': 'SYMMETRIC'},
    {'shape': [3], 'padding_matrix': [[0, 2]], 'mode': 'REFLECT'},

    {'shape': [3, 2, 4, 5], 'padding_matrix': [[1, 1], [2, 2], [1, 1], [1, 1]], 'mode': 'SYMMETRIC'},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteMirrorPadLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["MirrorPad"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'padding_matrix', 'mode'})) == 5, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            padding_matrix = tf.constant(np.array(params["padding_matrix"]))
            params['op_func'](tensor=place_holder, paddings=padding_matrix, mode=params['mode'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_mirror_pad(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
