import pytest
import tensorflow as tf
import numpy as np

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [1]},
    {'shape': [2, 3]},
    {'shape': [1, 1, 1, 1]},
    {'shape': [1, 3, 4, 3]},
    {'shape': [3, 15, 14, 3]},
]


class TestTFLiteHardSwishLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["HardSwish"]
    allowed_ops = ['HARD_SWISH']

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_dict['Input'] = np.float32((10 - (-10)) * np.random.random_sample(inputs_dict['Input']) + (-10))
        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape'})) == 1, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.float32, params['shape'],
                                                   name=self.inputs[0])
            hs = placeholder * tf.nn.relu6(placeholder + np.float32(3)) * np.float32(1. / 6.)
            tf.identity(hs, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_hardswish(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
