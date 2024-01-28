import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': ([random.randint(1, 4) for _ in range(1)], [random.randint(1, 4) for _ in range(1)])},
    {'shape': ([random.randint(1, 4) for _ in range(2)], [random.randint(1, 4) for _ in range(2)])},
    {'shape': ([random.randint(1, 4) for _ in range(3)], [random.randint(1, 4) for _ in range(3)])},
    {'shape': ([random.randint(1, 4) for _ in range(4)], [random.randint(1, 4) for _ in range(4)])},
    {'shape': ([random.randint(1, 4) for _ in range(4)], [random.randint(1, 4) for _ in range(4)])}
]


class TestTFLiteBatchMatmulLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["ZerosLike"]
    allowed_ops = ['MAXIMUM', 'ZEROS_LIKE']

    def _prepare_input(self, inputs_dict, generator=None):
        import numpy as np
        inputs_dict['Input'] = np.float32(np.random.randint(0, 100, self.shapes_set[1]))
        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape'})) == 1, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        self.shapes_set = params['shape']

        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=self.shapes_set[0], name=self.inputs[0])
            zeros = tf.zeros_like(placeholder)
            tf.maximum(zeros, placeholder, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_zeros_like(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
