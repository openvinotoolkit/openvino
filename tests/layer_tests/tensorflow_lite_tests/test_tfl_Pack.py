import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 22) for _ in range(1)], 'num_tensors': random.randint(2,3)},
    {'shape': [random.randint(1, 22) for _ in range(2)], 'num_tensors': random.randint(10,15)},
    {'shape': [random.randint(1, 22) for _ in range(3)], 'num_tensors': random.randint(5,7)},
    {'shape': [random.randint(1, 22) for _ in range(4)], 'num_tensors': random.randint(5,7)},
    {'shape': [random.randint(1, 22) for _ in range(5)], 'num_tensors': random.randint(7,8)},
]


class TestTFLitePackLayerTest(TFLiteLayerTest):
    inputs = []
    outputs = ["Pack"]
    allowed_ops = ['PACK']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'num_tensors'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
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
            tf.stack(tensors, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_pack(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
