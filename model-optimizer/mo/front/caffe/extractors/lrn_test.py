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

from mo.front.caffe.extractors.lrn import lrn_ext
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.utils.unittest.extractors import FakeMultiParam


class FakeProtoLayer:
    def __init__(self, val):
        self.lrn_param = val


class TestLRN(unittest.TestCase):
    def test_lrn_ext(self):
        params = {
            'alpha': 10,
            'beta': 15,
            'local_size': 20,
            'norm_region': 0
        }
        res = lrn_ext(FakeProtoLayer(FakeMultiParam(params)), None)
        exp_res = {
            'op': 'LRN',
            'type': 'LRN',
            'alpha': 10,
            'beta': 15,
            'local_size': 20,
            'region': 'across',
            'bias': 1,
            'infer': copy_shape_infer
        }
        self.assertEqual(res, exp_res)

    def test_lrn_ext_norm_reg(self):
        params = {
            'alpha': 10,
            'beta': 15,
            'local_size': 20,
            'norm_region': 1
        }
        res = lrn_ext(FakeProtoLayer(FakeMultiParam(params)), None)
        exp_res = {
            'op': 'LRN',
            'type': 'LRN',
            'alpha': 10,
            'beta': 15,
            'local_size': 20,
            'region': 'same',
            'bias': 1,
            'infer': copy_shape_infer
        }
        self.assertEqual(res, exp_res)
