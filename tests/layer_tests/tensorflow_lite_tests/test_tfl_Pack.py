import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['PACK'], 'op_func': tf.stack},
]

test_params = [
    {'shape': [3], 'num_tensors': 2},
    {'shape': [1, 22], 'num_tensors': 10},
    {'shape': [1, 1, 8], 'num_tensors': 5},
    {'shape': [1, 22, 22, 8], 'num_tensors': 5},
    {'shape': [1, 22, 22, 8, 3], 'num_tensors': 7},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLitePackLayerTest(TFLiteLayerTest):
    inputs = []
    outputs = ["Pack"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'num_tensors'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            tensors = []
            self.inputs = []
            for j in range(params['num_tensors']):
                name = f'Input_{j}'
                self.inputs.append(name)
                place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                        name=name)
                tensors.append(place_holder)
            params['op_func'](tensors, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_pack(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
