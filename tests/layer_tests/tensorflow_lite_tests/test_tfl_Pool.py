import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': 'AVERAGE_POOL_2D', 'op_func': 'tf.nn.avg_pool2d'},
    {'op_name': 'MAX_POOL_2D', 'op_func': 'tf.nn.max_pool2d'},
]

test_params = [
    {'shape': [1, 22, 22, 8], 'ksize': [3, 3], 'strides': 2, 'padding': 'SAME'},
    {'shape': [1, 22, 22, 8], 'ksize': [3, 3], 'strides': (2, 1), 'padding': 'SAME'},
    {'shape': [1, 22, 22, 8], 'ksize': [3, 3], 'strides': 2, 'padding': 'VALID'},
    {'shape': [1, 22, 22, 8], 'ksize': [3, 3], 'strides': (3, 4), 'padding': 'VALID'},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLitePoolLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Pool"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'ksize', 'strides',
                                                    'padding'})) == 6, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            eval(params['op_func'])(place_holder, params['ksize'], params['strides'],
                              params['padding'], 'NHWC', name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_pool(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
