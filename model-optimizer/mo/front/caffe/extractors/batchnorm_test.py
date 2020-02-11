"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.caffe.extractors.batchnorm import batch_norm_ext
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.utils.unittest.extractors import FakeParam, FakeModelLayer


class FakeBNProtoLayer:
    def __init__(self, eps):
        self.batch_norm_param = FakeParam('eps', eps)


class TestShapesParsing(unittest.TestCase):
    def test_bn_ext_no_ml_no_pb(self):
        self.assertRaises(AssertionError, batch_norm_ext, None, None)

    def test_bn_ext_no_ml(self):
        res = batch_norm_ext(FakeBNProtoLayer(10), None)
        exp_res = {
            'op': 'BatchNormalization',
            'type': 'BatchNormalization',
            'eps': 10,
            'infer': copy_shape_infer
        }
        self.assertEqual(res, exp_res)

    def test_bn_ext_ml_one_blob(self):
        self.assertRaises(AssertionError, batch_norm_ext, FakeBNProtoLayer(10), FakeModelLayer([np.array([1, 2])]))

    def test_bn_ext_ml_two_blobs(self):
        mean_blob = np.array([1., 2.])
        variance_blob = np.array([3., 4.])
        blobs = [mean_blob, variance_blob]
        res = batch_norm_ext(FakeBNProtoLayer(10),
                             FakeModelLayer(blobs))
        exp_res = {
            'type': 'BatchNormalization',
            'eps': 10,
            'infer': copy_shape_infer,
            'mean': mean_blob,
            'variance': variance_blob,
            'embedded_inputs': [
                (1, 'gamma', {
                    'bin': 'gamma'
                }),
                (2, 'beta', {
                    'bin': 'beta'
                }),
                (3, 'mean', {
                    'bin': 'biases'
                }),
                (4, 'variance', {
                    'bin': 'weights'
                })
            ]
        }
        for i in exp_res:
            if i in ('mean', 'variance'):
                np.testing.assert_array_equal(res[i], exp_res[i])
            else:
                self.assertEqual(res[i], exp_res[i])

    def test_bn_ext_ml_three_blobs(self):
        mean_blob = np.array([1., 2.])
        variance_blob = np.array([3., 4.])
        scale_blob = np.array([5., ])
        blobs = [mean_blob, variance_blob, scale_blob]
        res = batch_norm_ext(FakeBNProtoLayer(10),
                             FakeModelLayer(blobs))
        exp_res = {
            'type': 'BatchNormalization',
            'eps': 10,
            'infer': copy_shape_infer,
            'mean': mean_blob * 0.2,
            'variance': variance_blob * 0.2,
            'embedded_inputs': [
                (1, 'gamma', {
                    'bin': 'gamma'
                }),
                (2, 'beta', {
                    'bin': 'beta'
                }),
                (3, 'mean', {
                    'bin': 'biases'
                }),
                (4, 'variance', {
                    'bin': 'weights'
                })
            ]
        }
        for i in exp_res:
            if i in ('mean', 'variance'):
                np.testing.assert_array_equal(res[i], exp_res[i])
            else:
                self.assertEqual(res[i], exp_res[i])

    def test_bn_ext_ml_three_blobs_zero_scale(self):
        mean_blob = np.array([1., 2.])
        variance_blob = np.array([3., 4.])
        scale_blob = np.array([0., ])
        blobs = [mean_blob, variance_blob, scale_blob]
        res = batch_norm_ext(FakeBNProtoLayer(10),
                             FakeModelLayer(blobs))
        exp_res = {
            'type': 'BatchNormalization',
            'eps': 10,
            'infer': copy_shape_infer,
            'mean': mean_blob * 0.,
            'variance': variance_blob * 0.,
            'embedded_inputs': [
                (1, 'gamma', {
                    'bin': 'gamma'
                }),
                (2, 'beta', {
                    'bin': 'beta'
                }),
                (3, 'mean', {
                    'bin': 'biases'
                }),
                (4, 'variance', {
                    'bin': 'weights'
                })
            ]
        }
        for i in exp_res:
            if i in ('mean', 'variance'):
                np.testing.assert_array_equal(res[i], exp_res[i])
            else:
                self.assertEqual(res[i], exp_res[i])