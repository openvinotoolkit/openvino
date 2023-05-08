# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import itertools

import numpy as np
import pytest
import tensorflow as tf
from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

np.random.seed(42)


test_ops = [
    {'op_name': 'UNPACK', 'op_func': tf.unstack},
]

test_params = [
    {'shape': [3, 4, 3], 'axis': 0},
    {'shape': [3, 4, 3], 'axis': -1},
    {'shape': [3, 4, 3, 5], 'axis': -2},
    {'shape': [3, 4, 3, 5], 'axis': 2},
    {'shape': [1], 'axis': -1},
    {'shape': [1], 'axis': 0}
]

test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteUnpackLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Unpack"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'axis'})) == 4, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = [params['op_name']]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            params['op_func'](place_holder, axis=params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_unpack(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
