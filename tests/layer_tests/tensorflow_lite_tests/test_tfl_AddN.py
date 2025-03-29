import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

num_inputs = [
    {'num_inputs': 2},
    {'num_inputs': 4},
    {'num_inputs': 5},
]

test_params = [
    {'shape': [2, 10, 10, 3]},
    {'shape': [2, 10]}
]

test_data = parametrize_tests(num_inputs, test_params)


class TestTFLiteAddNLayerTest(TFLiteLayerTest):
    inputs = []
    outputs = ["AddN"]
    allowed_ops = ['ADD_N']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'num_inputs'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.inputs = []
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            inputs = []
            for j in range(params['num_inputs']):
                name = f"Input_{j}"
                self.inputs.append(name)
                placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                       name=name)
                inputs.append(placeholder)
            tf.math.add_n(inputs, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_add_n(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
