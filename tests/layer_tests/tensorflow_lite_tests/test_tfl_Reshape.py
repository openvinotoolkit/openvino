import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['RESHAPE'], 'op_func': tf.reshape},
]

test_params = [
    {'shape': [2, 6], 'out_shape': [2, 3, 2]},
    {'shape': [2, 4, 6], 'out_shape': [2, -1]},
    {'shape': [1], 'out_shape': []},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteReshapeLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Reshape"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'out_shape'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            out_shape = tf.constant(params['out_shape'], dtype=tf.int32)

            params['op_func'](place_holder, out_shape, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_reshape(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
