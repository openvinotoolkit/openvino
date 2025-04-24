import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [1, 5], 'broadcast_shape': [5, 5]},
    {'shape': [1], 'broadcast_shape': [7]}
]


class TestTFLiteBroadcastArgsLayerTest(TFLiteLayerTest):
    inputs = ["Input", "Input1"]
    outputs = ["BroadcastArgsOP"]
    allowed_ops = ['BROADCAST_ARGS']

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_dict["Input"] = np.array(self.shape).astype(np.int32)
        inputs_dict["Input1"] = np.array(self.broadcast_shape).astype(np.int32)
        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'broadcast_shape'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        self.shape = params['shape']
        self.broadcast_shape = params['broadcast_shape']

        with tf.compat.v1.Session() as sess:
            s0 = tf.compat.v1.placeholder(params.get('dtype', tf.int32), [len(self.shape)],
                                          name=self.inputs[0])
            s1 = tf.compat.v1.placeholder(params.get('dtype', tf.int32), [len(self.shape)],
                                          name=self.inputs[1])
            tf.raw_ops.BroadcastArgs(s0=s0, s1=s1, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_broadcast_args(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
