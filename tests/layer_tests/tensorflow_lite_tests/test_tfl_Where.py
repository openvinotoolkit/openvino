import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'shape': [10]},
    {'shape': [1, 2, 3, 4]},
    {'shape': [8, 7, 6, 5, 4, 3, 2, 1]}
]


class TestTFLiteWhereLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Where"]
    allowed_ops = ['WHERE']

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_dict['Input'] = np.random.randint(0, 1, inputs_dict['Input']) < 1
        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape'})) == 1, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            input_value1 = tf.compat.v1.placeholder(tf.bool, params['shape'], name=self.inputs[0])
            tf.where(input_value1, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_where(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
