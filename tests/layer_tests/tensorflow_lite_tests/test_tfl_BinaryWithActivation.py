import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import additional_test_params, activation_helper
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': 'ADD', 'op_func': 'tf.math.add'},
    {'op_name': 'DIV', 'op_func': 'tf.math.divide', 'kwargs_to_prepare_input': 'positive'},
    {'op_name': 'MUL', 'op_func': 'tf.math.multiply'},
    {'op_name': 'SUB', 'op_func': 'tf.math.subtract'},
]

test_params = [
    {'shape': [2, 10, 10, 3]},
    {'shape': [2, 10]}
]

test_data = parametrize_tests(test_ops, test_params)
test_data = parametrize_tests(test_data, additional_test_params[1])


class TestTFLiteBinaryWithActivationLayerTest(TFLiteLayerTest):
    inputs = ["Input_0", "Input_1"]
    outputs = ["BinaryOperation"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'activation'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            in0 = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'], name=self.inputs[0])
            in1 = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'], name=self.inputs[1])
            bin_op_name = self.outputs[0] if not params['activation'] else \
                self.outputs[0] + "/op"
            op = eval(params['op_func'])(in0, in1, name=bin_op_name)
            op = activation_helper(op, eval(params['activation']) if params['activation'] else None, self.outputs[0])

            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_binary(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
