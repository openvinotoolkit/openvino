import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'op_name': 'PAD', 'shape': [random.randint(1, 10) for _ in range(5)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]]},
    {'op_name': 'PAD', 'shape': [random.randint(1, 10) for _ in range(5)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]]},

    {'op_name': 'PAD', 'shape': [random.randint(1, 10) for _ in range(4)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]]},
    {'op_name': 'PAD', 'shape': [random.randint(1, 10) for _ in range(4)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]]},

    {'op_name': 'PAD', 'shape': [random.randint(1,3), random.randint(1,2)], 'paddings': [[random.randint(1,2), random.randint(1,2)], [random.randint(1,2), random.randint(1,2)]]},
    {'op_name': 'PAD', 'shape': [random.randint(1, 10) for _ in range(2)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]]},

    {'op_name': 'PAD', 'shape': [random.randint(1, 10) for _ in range(1)], 'paddings': [[random.randint(1, 10) for _ in range(2)]]},

    {'op_name': 'PADV2', 'shape': [random.randint(1, 10) for _ in range(4)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]],
     'constant_value': -1},
    {'op_name': 'PADV2', 'shape': [random.randint(1, 10) for _ in range(4)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]],
     'constant_value': 1},

    {'op_name': 'PADV2', 'shape': [random.randint(1, 10) for _ in range(4)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]], 'constant_value': -1},
    {'op_name': 'PADV2', 'shape': [random.randint(1, 10) for _ in range(4)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]], 'constant_value': 1},

    {'op_name': 'PADV2', 'shape': [random.randint(1, 10) for _ in range(2)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]], 'constant_value': 1},
    {'op_name': 'PADV2', 'shape': [random.randint(1, 10) for _ in range(2)], 'paddings': [[random.randint(1, 10) for _ in range(2)], [random.randint(1, 10) for _ in range(2)]], 'constant_value': -1},

    {'op_name': 'PADV2', 'shape': [random.randint(1, 10) for _ in range(1)], 'paddings': [[random.randint(1, 10) for _ in range(2)]], 'constant_value': -1},

]


class TestTFLitePadLayerTest(TFLiteLayerTest):
    outputs = ["Pad"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'shape', 'paddings'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        self.allowed_ops = [params['op_name']]
        self.inputs = ["Input"]
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(tf.float32, params['shape'], name=self.inputs[0])
            if params['op_name'] == 'PADV2':
                tf.pad(tensor=place_holder, paddings=params['paddings'], constant_values=params['constant_value'],
                       name=self.outputs[0])
            else:
                self.inputs.append('Paddings')
                shape = [len(params["paddings"]), 2]
                paddings = tf.compat.v1.placeholder(dtype=tf.int32, name=self.inputs[1], shape=shape)
                tf.pad(tensor=place_holder, paddings=paddings, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_pad(self, params, ie_device, precision, temp_dir):
        if params['op_name'] == 'PAD':
            pytest.xfail("CVS-110828")
        self._test(ie_device, precision, temp_dir, params)
