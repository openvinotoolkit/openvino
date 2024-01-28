import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 10) for _ in range(4)], 'num_or_size_splits': random.randint(1,2), 'axis': random.randint(0,1)},
    {'shape': [random.randint(1, 10) for _ in range(3)], 'num_or_size_splits': random.randint(1,4), 'axis': random.randint(1,3)},
    {'shape': [random.randint(1, 10) for _ in range(4)], 'num_or_size_splits': random.randint(1,5), 'axis': random.randint(-4,-1)},
]


class TestTFLiteSplitLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Split"]
    allowed_ops = ['SPLIT']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'num_or_size_splits',
                                                    'axis'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            tf.split(placeholder, params["num_or_size_splits"], params["axis"], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_split(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
