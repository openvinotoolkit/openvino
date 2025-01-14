# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'shape': [15]},
    {'shape': [1]}
]


class TestTFLiteUniqueLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Unqiue"]
    allowed_ops = ['UNIQUE']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape'})) == 1, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                   name=self.inputs[0])
            tf.unique(placeholder, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_unique(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
