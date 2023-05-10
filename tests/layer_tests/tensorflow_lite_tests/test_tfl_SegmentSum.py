import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [2, 3, 1, 2, 2], 'segment_ids': [0, 1, 2, 3, 4]},
    {'shape': [2, 3, 1, 2], 'segment_ids': [0, 1, 2, 3]},
    {'shape': [2, 3], 'segment_ids': [0, 1]},
]


class TestTFLiteSegmentSumLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["SegmentSum"]
    allowed_ops = ['SEGMENT_SUM']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'segment_ids'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            constant = tf.constant(params['segment_ids'], dtype=tf.int32)
            tf.math.segment_sum(place_holder, constant, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_segment_sum(self, params, ie_device, precision, temp_dir):
        if params['shape'] == [2, 3, 1, 2, 2] or [2, 3, 1, 2]:
            pytest.xfail('CVS-110478')
        self._test(ie_device, precision, temp_dir, params)
