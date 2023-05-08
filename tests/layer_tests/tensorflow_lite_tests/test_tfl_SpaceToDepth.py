import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': 'SPACE_TO_DEPTH', 'op_func': tf.nn.space_to_depth},
]

test_params = [
    {'shape': [8, 10, 10, 16], 'block_size': 2, 'data_format': 'NHWC'},
    {'shape': [24, 10, 10, 50], 'block_size': 5, 'data_format': 'NHWC'},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteSpaceToDepthLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["SpaceToDepth"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'block_size', 'data_format'})) == 5, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            params['op_func'](place_holder, params['block_size'], params['data_format'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_space_to_depth(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
