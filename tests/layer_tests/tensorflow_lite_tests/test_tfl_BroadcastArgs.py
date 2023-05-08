import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': 'BROADCAST_ARGS', 'op_func': tf.raw_ops.BroadcastArgs},
]

test_params = [
    {'shape': [1, 5], 'broadcast_shape': [5, 5]},
    {'shape': [1], 'broadcast_shape': [7]}
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteBroadcastArgsLayerTest(TFLiteLayerTest):
    inputs = ["Input", "Input1"]
    outputs = ["BroadcastArgsOP"]

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_dict["Input"] = np.array(self.shape).astype(np.int32)
        inputs_dict["Input1"] = np.array(self.broadcast_shape).astype(np.int32)
        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'broadcast_shape'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        self.shape = params['shape']
        self.broadcast_shape = params['broadcast_shape']

        with tf.compat.v1.Session() as sess:
            s0 = tf.compat.v1.placeholder(params.get('dtype', tf.int32), [len(self.shape)],
                                          name=self.inputs[0])
            s1 = tf.compat.v1.placeholder(params.get('dtype', tf.int32), [len(self.shape)],
                                          name=self.inputs[1])
            params['op_func'](s0=s0, s1=s1, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_broadcast_args(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
