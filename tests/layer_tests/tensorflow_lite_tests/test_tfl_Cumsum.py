# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'input_shape': [2], 'axis':-1, 'exclusive': False, 'reverse':  False},
    {'input_shape': [2, 3], 'axis':0, 'exclusive': False, 'reverse':  False},
    {'input_shape': [2, 3], 'axis':0, 'exclusive': True, 'reverse':  False},
    {'input_shape': [2, 3], 'axis':0, 'exclusive': False, 'reverse':  True},
]


class TestTFLiteCumsum(TFLiteLayerTest):
    # input_shape - should be an array
    # axis - array which points on axis for the operation
    # exclusive - enables exclusive Cumsum
    # reverse - enables reverse order of Cumsum
    inputs = ["Input"]
    outputs = ['Cumsum']
    allowed_ops = ['CUMSUM']

    def make_model(self, params):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32),
                                                             params['input_shape'],
                                                             name=self.inputs[0])
            tf.cumsum(placeholder, params['axis'],  params['exclusive'], 
                          params['reverse'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_cumsum(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
