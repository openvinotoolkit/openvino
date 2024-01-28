import platform
import random
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 10) for _ in range(1)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(1)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(1)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(1)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},

    {'shape': [random.randint(1, 10) for _ in range(2)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(2)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(2)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(2)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},

    {'shape': [random.randint(1, 10) for _ in range(3)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(3)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(3)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(3)], 'segment_ids': [random.randint(0, 10) for _ in range(4)]},
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
