import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['PAD'], 'op_func': tf.pad},
]

test_params = [
    {'shape': [1, 1, 2, 1, 1], 'paddings': [[0, 0], [0, 1], [2, 3], [0, 0], [1, 0]]},
    {'shape': [2, 1, 1, 1, 1], 'paddings': [[0, 1], [0, 0], [0, 0], [2, 3], [1, 0]]},

    {'shape': [1, 1, 2, 1], 'paddings': [[0, 0], [0, 1], [2, 3], [0, 0]]},
    {'shape': [1, 1, 2, 1], 'paddings': [[0, 0], [0, 1], [2, 3], [0, 0]]},

    {'shape': [1, 2], 'paddings': [[0, 1], [2, 1]]},
    {'shape': [1, 2], 'paddings': [[2, 3], [0, 1]]},

    {'shape': [1], 'paddings': [[1, 2]]},

]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLitePadLayerTest(TFLiteLayerTest):
    inputs = ["Input", 'Paddings']
    outputs = ["Pad"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'paddings'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            shape = [len(params["paddings"]), 2]
            paddings = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[1], shape=shape)
            params['op_func'](tensor=place_holder, paddings=paddings, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_pad(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
