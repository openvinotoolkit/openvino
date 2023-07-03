import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [3], 'num_or_size_splits': [0, 2, 0, 1], 'axis': 0},
    {'shape': [2, 3, 9], 'num_or_size_splits': [1, 2, 3, -1, 1], 'axis': 2},
    {'shape': [3, 9, 5, 4], 'num_or_size_splits': [1, 2, 0, -1, 2, 0], 'axis': -3},
]


class TestTFLiteSplitLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["SplitV"]
    allowed_ops = ['SPLIT_V']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'num_or_size_splits',
                                                    'axis'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            size_splits = tf.constant(params['num_or_size_splits'], dtype=tf.int32)
            axis = tf.constant(params['axis'], dtype=tf.int32)
            num_split = len(params['num_or_size_splits'])

            tf.raw_ops.SplitV(value=placeholder, size_splits=size_splits, axis=axis,
                              num_split=num_split, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_split_v(self, params, ie_device, precision, temp_dir):
        if 0 in params['num_or_size_splits']:
            pytest.skip("CVS-110040")
        self._test(ie_device, precision, temp_dir, params)
