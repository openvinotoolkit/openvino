import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 10) for _ in range(1)], 'axis': [random.randint(-3, 7) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(1)], 'axis': [random.randint(-3, 7) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(2)], 'axis': [random.randint(-3, 7) for _ in range(2)]},
    {'shape': [random.randint(1, 10) for _ in range(2)], 'axis': [random.randint(-3, 7) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(3)], 'axis': [random.randint(-3, 7) for _ in range(2)]},
    {'shape': [random.randint(1, 10) for _ in range(3)], 'axis': [random.randint(-3, 7) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(4)], 'axis': [random.randint(-3, 7) for _ in range(4)]},
    {'shape': [random.randint(1, 10) for _ in range(4)], 'axis': [random.randint(-3, 7) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(5)], 'axis': [random.randint(-3, 7) for _ in range(5)]},
    {'shape': [random.randint(1, 10) for _ in range(5)], 'axis': [random.randint(-3, 7) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(8)], 'axis': [random.randint(-3, 7) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(8)], 'axis': [random.randint(-3, 7) for _ in range(8)]},
]


class TestTFLiteReverseV2LayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["ReverseV2"]
    allowed_ops = ['REVERSE_V2']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'axis'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.reverse(place_holder, params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_reverse_v2(self, params, ie_device, precision, temp_dir):
        if len(params['axis']) > 1:
            pytest.xfail('CVS-109932')
        self._test(ie_device, precision, temp_dir, params)
