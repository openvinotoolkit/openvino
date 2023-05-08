# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

np.random.seed(42)

test_ops = [
    {'op_name': 'UNIQUE', 'op_func': tf.unique},
]

test_params = [
    {'shape': [15]},
    {'shape': [1]}
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteUniqueLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Unqiue"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                   name=self.inputs[0])
            params['op_func'](placeholder, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_unique(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
