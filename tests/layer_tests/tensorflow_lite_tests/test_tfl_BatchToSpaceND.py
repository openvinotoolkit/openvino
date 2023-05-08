import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': 'BATCH_TO_SPACE_ND', 'op_func': tf.batch_to_space},
]

test_params = [
    {'shape': [8, 10, 10, 3], 'block_shape': [2, 2], 'crops': [[0, 2], [0, 0]]},
    {'shape': [24, 10, 10, 1], 'block_shape': [2, 12], 'crops': [[2, 0], [0, 2]]}
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteBatchToSpaceNDLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["BatchToSpaceND"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'block_shape', 'crops'})) == 5, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            params['op_func'](place_holder, params['block_shape'], params['crops'],
                              name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_batch_to_space_nd(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
