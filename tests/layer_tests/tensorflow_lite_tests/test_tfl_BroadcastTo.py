import platform
import random
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 40) for _ in range(5)], 'broadcast_shape': [random.randint(1, 40) for _ in range(5)]},
    {'shape': [random.randint(1, 40) for _ in range(5)], 'broadcast_shape': [random.randint(1, 40) for _ in range(5)]}
]


class TestTFLiteBroadcastToLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["BroadcastToOP"]
    allowed_ops = ['BROADCAST_TO']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'broadcast_shape'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            shape_of = tf.reshape(place_holder, params['shape'])
            tf.broadcast_to(input=shape_of, shape=params['broadcast_shape'],
                            name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 123324')
    def test_broadcast_to(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
