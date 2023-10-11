import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'op_name': 'PAD', 'shape': [1, 1, 2, 1, 1], 'paddings': [[0, 0], [0, 1], [2, 3], [0, 0], [1, 0]]},
    {'op_name': 'PAD', 'shape': [2, 1, 1, 1, 1], 'paddings': [[0, 1], [0, 0], [0, 0], [2, 3], [1, 0]]},

    {'op_name': 'PAD', 'shape': [1, 1, 2, 1], 'paddings': [[0, 0], [0, 1], [2, 3], [0, 0]]},
    {'op_name': 'PAD', 'shape': [1, 1, 2, 1], 'paddings': [[0, 0], [0, 1], [2, 3], [0, 0]]},

    {'op_name': 'PAD', 'shape': [1, 2], 'paddings': [[0, 1], [2, 1]]},
    {'op_name': 'PAD', 'shape': [1, 2], 'paddings': [[2, 3], [0, 1]]},

    {'op_name': 'PAD', 'shape': [1], 'paddings': [[1, 2]]},

    {'op_name': 'PADV2', 'shape': [1, 1, 2, 1, 1], 'paddings': [[0, 0], [0, 1], [2, 3], [0, 0], [1, 0]],
     'constant_value': -1},
    {'op_name': 'PADV2', 'shape': [2, 1, 1, 1, 1], 'paddings': [[0, 1], [0, 0], [0, 0], [2, 3], [1, 0]],
     'constant_value': 1},

    {'op_name': 'PADV2', 'shape': [1, 1, 2, 1], 'paddings': [[0, 0], [0, 1], [2, 3], [0, 0]], 'constant_value': -1},
    {'op_name': 'PADV2', 'shape': [1, 1, 2, 1], 'paddings': [[0, 0], [0, 1], [2, 3], [0, 0]], 'constant_value': 1},

    {'op_name': 'PADV2', 'shape': [1, 2], 'paddings': [[0, 1], [2, 1]], 'constant_value': 1},
    {'op_name': 'PADV2', 'shape': [1, 2], 'paddings': [[2, 3], [0, 1]], 'constant_value': -1},

    {'op_name': 'PADV2', 'shape': [1], 'paddings': [[1, 2]], 'constant_value': -1},

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
