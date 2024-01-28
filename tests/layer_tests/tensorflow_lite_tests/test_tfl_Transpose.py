import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 4) for _ in range(2)], 'perm': [random.randint(1, 4) for _ in range(2)], 'conjugate': random.choice([True, False])},
    {'shape': [random.randint(1, 4) for _ in range(3)], 'perm': [random.randint(1, 4) for _ in range(3)], 'conjugate': random.choice([True, False])},
    {'shape': [random.randint(1,2), random.randint(1,3), random.randint(1,5), random.randint(1,10)], 'perm': [random.randint(1,2), random.randint(1,2), random.randint(1,3), random.randint(0,1)], 'conjugate': random.choice([True, False])},
]


class TestTFLiteTransposeLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Transpose"]
    allowed_ops = ['TRANSPOSE']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'perm', 'conjugate'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            tf.transpose(placeholder, params['perm'], params['conjugate'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_transpose(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
