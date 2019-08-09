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

from mo.front.caffe.extractors.inner_product import inner_product_ext
from mo.front.common.partial_infer.inner_product import caffe_inner_product
from mo.utils.unittest.extractors import FakeMultiParam, FakeModelLayer


class FakeProtoLayer:
    def __init__(self, val):
        self.inner_product_param = val


class TestInnerProduct(unittest.TestCase):
    def test_inner_product_ext(self):
        params = {
            'num_output': 10,
            'bias_term': True
        }
        mean_blob = np.array([1., 2.])
        variance_blob = np.array([3., 4.])
        blobs = [mean_blob, variance_blob]
        res = inner_product_ext(FakeProtoLayer(FakeMultiParam(params)),
                                FakeModelLayer(blobs))
        exp_res = {
            'type': 'MatMul',
            'out-size': 10,
            'infer': caffe_inner_product,
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
