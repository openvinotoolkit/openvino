import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 10) for _ in range(2)], 'out_shape': [random.randint(1, 10) for _ in range(3)]},
    {'shape': [random.randint(1, 10) for _ in range(3)], 'out_shape': [random.randint(-4, 10) for _ in range(2)]},
    {'shape': [random.randint(1, 10) for _ in range(1)], 'out_shape': [random.randint(1, 10) for _ in range(4)]},
]


class TestTFLiteReshapeLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Reshape"]
    allowed_ops = ['RESHAPE']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'out_shape'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            out_shape = tf.constant(params['out_shape'], dtype=tf.int32)

            tf.reshape(place_holder, out_shape, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_reshape(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
