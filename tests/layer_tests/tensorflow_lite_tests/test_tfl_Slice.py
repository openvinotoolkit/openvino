import string

import numpy as np
import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    # OV doesn't support string format
    # {'shape': [12, 2, 2, 5], 'begin': [0, 0, 0, 0], 'size': [8, 2, 2, 3], 'dtype': tf.string},
    {'shape': [random.randint(2, 15) for _ in range(5)], 'begin': [random.randint(0, 5) for _ in range(4)], 'size': [random.randint(1, 15) for _ in range(4)], 'dtype': np.float32},
    {'shape': [random.randint(1, 5) for _ in range(4)], 'begin': [random.randint(0, 5) for _ in range(4)], 'size': [random.randint(-1, 5) for _ in range(4)], 'dtype': np.float32},
    {'shape': [random.randint(1, 6) for _ in range(5)], 'begin': [random.randint(1, 5) for _ in range(5)], 'size': [random.randint(1, 5) for _ in range(5)], 'dtype': np.int32},
    {'shape': [random.randint(1, 5) for _ in range(2)], 'begin': [1, 0], 'size': [-1, -1], 'dtype': np.int64},
]


class TestTFLiteSliceLayerTest(TFLiteLayerTest):
    inputs = ["Input", "Begin", "Size"]
    outputs = ["Slice"]
    allowed_ops = ['SLICE']

    def _prepare_input(self, inputs_dict, generator=None):
        if self.input_dtype == tf.string:
            letters = list(string.ascii_uppercase)
            inputs_dict["Input"] = np.random.choice(letters, size=self.shape).astype(self.input_dtype)
        else:
            inputs_dict["Input"] = np.random.randint(-1, 2, self.shape).astype(self.input_dtype)

        inputs_dict["Begin"] = np.array(self.begin).astype(np.int32)
        inputs_dict['Size'] = np.array(self.size).astype(np.int32)

        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'begin', 'size', 'dtype'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.input_dtype = params['dtype']
        self.shape = params['shape']
        self.begin = params['begin']
        self.size = params['size']

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params['dtype'], params['shape'], self.inputs[0])
            begin = tf.compat.v1.placeholder(tf.int32, len(params['shape']), name=self.inputs[1])
            size = tf.compat.v1.placeholder(tf.int32, len(params['shape']), name=self.inputs[2])

            tf.slice(placeholder, begin, size, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_slice(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
