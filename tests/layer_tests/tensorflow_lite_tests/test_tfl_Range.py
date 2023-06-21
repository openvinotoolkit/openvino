import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'dtype': np.float32, 'negative_delta': False},
    {'dtype': np.int32, 'negative_delta': True},
]


class TestTFLiteRangeLayerTest(TFLiteLayerTest):
    inputs = ['Start', 'Limit', 'Delta']
    outputs = ["Range"]
    allowed_ops = ['RANGE']

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_data = {}
        if self.negative_delta:
            inputs_data['Start'] = np.random.randint(1, 10, []).astype(self.dtype)
            inputs_data['Limit'] = np.random.randint(-10, 0, []).astype(self.dtype)
            inputs_data['Delta'] = np.random.randint(-5, -1, []).astype(self.dtype)
        else:
            inputs_data['Start'] = np.random.randint(1, 10, []).astype(self.dtype)
            inputs_data['Limit'] = np.random.randint(10, 30, []).astype(self.dtype)
            inputs_data['Delta'] = np.random.randint(1, 5, []).astype(self.dtype)

        return inputs_data

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'dtype', 'negative_delta'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            self.dtype = params['dtype']
            self.negative_delta = params['negative_delta']

            start = tf.compat.v1.placeholder(self.dtype, [], self.inputs[0])
            limit = tf.compat.v1.placeholder(self.dtype, [], self.inputs[1])
            delta = tf.compat.v1.placeholder(self.dtype, [], self.inputs[2])

            tf.range(start, limit, delta, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_range(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
