"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

import numpy as np

from mo.front.caffe.extractors.scale import scale_ext
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.utils.unittest.extractors import FakeMultiParam, FakeModelLayer


class FakeProtoLayer:
    def __init__(self, val, bottom2=False):
        self.scale_param = val
        if bottom2:
            self.bottom = {"bottom1", "bottom2"}
        else:
            self.bottom = {"bottom1"}


class TestScale(unittest.TestCase):
    def test_scale_ext(self):
        mean_blob = np.array([1., 2.])
        variance_blob = np.array([3., 4.])
        blobs = [mean_blob, variance_blob]
        params = {
            'type': 'Scale',
            'axis': 0,
            'bias_term': True
        }

        res = scale_ext(FakeProtoLayer(FakeMultiParam(params)), FakeModelLayer(blobs))
        exp_res = {
            'op': 'ScaleShift',
            'type': 'ScaleShift',
            'axis': 0,
            'infer': copy_shape_infer,
            'weights': mean_blob,
            'biases': variance_blob,
            'embedded_inputs': [
                (1, 'weights', {
                    'bin': 'weights'
                }),
                (2, 'biases', {
                    'bin': 'biases'
                })
            ]
        }
        for i in exp_res:
            if i in ('weights', 'biases'):
                np.testing.assert_array_equal(res[i], exp_res[i])
            else:
                self.assertEqual(res[i], exp_res[i])

    def test_scale_2inputs_ext(self):
        params = {
            'type': 'Scale',
            'axis': 0,
            'bias_term': False
        }

        res = scale_ext(FakeProtoLayer(FakeMultiParam(params), True), None)
        exp_res = {
            'op': 'ScaleShift',
            'type': 'ScaleShift',
            'axis': 0,
            'infer': copy_shape_infer,
        }
        for i in exp_res:
            self.assertEqual(res[i], exp_res[i])

    def test_scale_2inputs_bias_ext(self):
        variance_blob = np.array([3., 4.])
        blobs = [variance_blob]

        params = {
            'type': 'Scale',
            'axis': 0,
            'bias_term': True
        }

        res = scale_ext(FakeProtoLayer(FakeMultiParam(params), True), FakeModelLayer(blobs))
        exp_res = {
            'op': 'ScaleShift',
            'type': 'ScaleShift',
            'axis': 0,
            'infer': copy_shape_infer,
            'biases': variance_blob,
            'embedded_inputs': [
                (1, 'biases', {
                    'bin': 'biases'
                })]
        }
        for i in exp_res:
            if i in ('biases'):
                np.testing.assert_array_equal(res[i], exp_res[i])
            else:
                self.assertEqual(res[i], exp_res[i])

    def test_create_default_weights(self):
        """
        There are situations when scale layer doesn't have weights and biases. This test checks that if they are not
        available in the caffemodel file then default values [1] and [0] are generated.
        """
        scale_blob = np.array([1])
        bias_blob = np.array([0])
        params = {
            'type': 'Scale',
            'axis': 0,
            'bias_term': True
        }

        res = scale_ext(FakeProtoLayer(FakeMultiParam(params)), None)
        exp_res = {
            'op': 'ScaleShift',
            'type': 'ScaleShift',
            'axis': 0,
            'infer': copy_shape_infer,
            'weights': scale_blob,
            'biases': bias_blob,
            'embedded_inputs': [
                (1, 'weights', {
                    'bin': 'weights'
                }),
                (2, 'biases', {
                    'bin': 'biases'
                })
            ]
        }
        self.assertDictEqual(exp_res, res)
