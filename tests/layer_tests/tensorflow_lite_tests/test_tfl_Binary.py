import platform
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': 'EQUAL', 'op_func': 'tf.math.equal'},
    {'op_name': 'FLOOR_MOD', 'op_func': 'tf.math.floormod', 'kwargs_to_prepare_input': 'positive'},
    {'op_name': 'FLOOR_DIV', 'op_func': 'tf.math.floordiv', 'kwargs_to_prepare_input': 'positive'},
    {'op_name': 'GREATER', 'op_func': 'tf.math.greater'},
    {'op_name': 'GREATER_EQUAL', 'op_func': 'tf.math.greater_equal'},
    {'op_name': 'LESS', 'op_func': 'tf.math.less'},
    {'op_name': 'LESS_EQUAL', 'op_func': 'tf.math.less_equal'},
    {'op_name': 'LOGICAL_AND', 'op_func': 'tf.math.logical_and', 'kwargs_to_prepare_input': 'boolean', 'dtype': tf.bool},
    {'op_name': 'LOGICAL_OR', 'op_func': 'tf.math.logical_or', 'kwargs_to_prepare_input': 'boolean', 'dtype': tf.bool},
    {'op_name': 'MAXIMUM', 'op_func': 'tf.math.maximum'},
    {'op_name': 'MINIMUM', 'op_func': 'tf.math.minimum'},
    {'op_name': 'NOT_EQUAL', 'op_func': 'tf.math.not_equal'},
    {'op_name': 'POW', 'op_func': 'tf.math.pow', 'kwargs_to_prepare_input': 'positive'},
    {'op_name': 'SQUARED_DIFFERENCE', 'op_func': 'tf.math.squared_difference'},
]

test_params = [
    {'shape': [2, 10, 10, 3]},
    {'shape': [2, 10]}
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteBinaryLayerTest(TFLiteLayerTest):
    inputs = ["Input_0", "Input_1"]
    outputs = ["BinaryOperation"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder0 = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                     name=self.inputs[0])
            place_holder1 = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                     name=self.inputs[1])
            eval(params['op_func'])(place_holder0, place_holder1, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.xfail(platform.machine() in ["aarch64", "arm64", "ARM64"],
                       reason='Ticket - 123324')
    def test_binary(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
