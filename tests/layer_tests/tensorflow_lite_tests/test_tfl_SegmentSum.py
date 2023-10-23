import platform

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [4], 'segment_ids': [0, 0, 1, 1]},
    {'shape': [4], 'segment_ids': [0, 1, 2, 2]},
    {'shape': [4], 'segment_ids': [0, 1, 2, 3]},
    {'shape': [4], 'segment_ids': [0, 0, 0, 0]},

    {'shape': [4, 4], 'segment_ids': [0, 0, 1, 1]},
    {'shape': [4, 4], 'segment_ids': [0, 1, 2, 2]},
    {'shape': [4, 4], 'segment_ids': [0, 1, 2, 3]},
    {'shape': [4, 4], 'segment_ids': [0, 0, 0, 0]},

    {'shape': [4, 3, 2], 'segment_ids': [0, 0, 0, 0]},
    {'shape': [4, 3, 2], 'segment_ids': [0, 0, 1, 1]},
    {'shape': [4, 3, 2], 'segment_ids': [0, 1, 2, 2]},
    {'shape': [4, 3, 2], 'segment_ids': [0, 1, 2, 3]},
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
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 123324')
    def test_segment_sum(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
