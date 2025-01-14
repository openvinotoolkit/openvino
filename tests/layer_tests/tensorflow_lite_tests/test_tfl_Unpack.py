# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'shape': [3, 4, 3], 'axis': 0},
    {'shape': [3, 4, 3], 'axis': -1},
    {'shape': [3, 4, 3, 5], 'axis': -2},
    {'shape': [3, 4, 3, 5], 'axis': 2},
    {'shape': [1], 'axis': -1},
    {'shape': [1], 'axis': 0}
]


class TestTFLiteUnpackLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Unpack"]
    allowed_ops = ['UNPACK']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'axis'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.unstack(place_holder, axis=params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_unpack(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
