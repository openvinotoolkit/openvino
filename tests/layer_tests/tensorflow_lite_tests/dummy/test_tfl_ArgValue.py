import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import additional_test_params
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': 'ARG_MAX', 'op_func': tf.math.argmax},
    {'op_name': 'ARG_MIN', 'op_func': tf.math.argmin},
]

test_params = [
    {'shape': [random.randint(1, 4) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(2)]}
]

test_data = parametrize_tests(test_ops, test_params)
test_data = parametrize_tests(test_data, additional_test_params[0])


class TestTFLiteArgValueLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["ArgValue"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'axis'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            params['op_func'](place_holder, axis=params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_arg_value(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
