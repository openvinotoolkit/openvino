import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['SEGMENT_SUM'], 'op_func': tf.math.segment_sum},
]

test_params = [
    {'shape': [2, 3, 1, 2, 2], 'segment_ids': [0, 1, 2, 3, 4]},
    {'shape': [2, 3, 1, 2], 'segment_ids': [-1, -2, -3, -4]},
    {'shape': [2, 3], 'segment_ids': [0, 1]},
    {'shape': [2], 'segment_ids': [-1]},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteSegmentSumLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["SegmentSum"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'segment_ids'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            constant = tf.constant(params['segment_ids'], dtype=tf.int32)
            params['op_func'](place_holder, constant, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_segment_sum(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
