import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape_x': [random.randint(1, 40) for _ in range(2)], 'shape_y': [random.randint(1, 40) for _ in range(2)]},
    {'shape_x': [random.randint(1, 10) for _ in range(2)], 'shape_y': [random.randint(1, 10) for _ in range(2)]},
    {'shape_x': [random.randint(1, 10) for _ in range(3)], 'shape_y': [random.randint(1, 10) for _ in range(2)]},
]


class TestTFLiteFullyConnectedLayerTest(TFLiteLayerTest):
    inputs = ["Input_x", "Input_y"]
    outputs = ["FullyConnected"]
    allowed_ops = (['FULLY_CONNECTED'], ['BATCH_MATMUL'])

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape_x', 'shape_y'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape_x'], name=self.inputs[0])
            y = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape_y'], name=self.inputs[1])

            tf.matmul(x, y, transpose_a=False, transpose_b=True, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_fully_connected(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
