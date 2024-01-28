import pytest
import tensorflow as tf
import numpy as np
import random	
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 10) for _ in range(1)], 'axis': [random.randint(0, 10) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(2)], 'axis': [random.randint(1, 10) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(2)], 'axis': [random.randint(0, 10) for _ in range(2)]},
    {'shape': [random.randint(1, 10) for _ in range(4)], 'axis': [random.randint(1, 10) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(4)], 'axis': [random.randint(1, 10) for _ in range(2)]},
    {'shape': [random.randint(1, 10) for _ in range(4)], 'axis': [random.randint(-3,-1) for _ in range(3)]},
]


class TestTFLiteExpandDimsLayerTest(TFLiteLayerTest):
    inputs = ["Input", "Input1"]
    outputs = ["ExpandDims"]
    allowed_ops = ['EXPAND_DIMS']

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_dict['Input'] = (1.0 - (-1.0)) * np.random.random_sample(inputs_dict['Input']) + (-1.0)
        inputs_dict['Input1'] = self.axis

        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'axis'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        self.axis = params['axis']

        with tf.compat.v1.Session() as sess:
            input_value1 = tf.compat.v1.placeholder(tf.float32, params['shape'], name=self.inputs[0])
            axis = tf.compat.v1.placeholder(tf.int32, [len(params['axis'])], name=self.inputs[1])

            tf.expand_dims(input_value1, axis, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_expand_dims(self, params, ie_device, precision, temp_dir):
        pytest.xfail("CVS-111983")
        self._test(ie_device, precision, temp_dir, params)
