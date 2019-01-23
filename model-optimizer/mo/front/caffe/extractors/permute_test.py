"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.caffe.extractors.permute import permute_ext
from mo.front.common.partial_infer.transpose import transpose_infer
from mo.utils.unittest.extractors import FakeMultiParam


class FakePermuteProtoLayer:
    def __init__(self, val):
        self.permute_param = val


class TestPermuteParsing(unittest.TestCase):
    def test_permute_check_attrs(self):
        attrs = {
            'order': np.array([0, 1, 3, 2])
        }

        res = permute_ext(FakePermuteProtoLayer(FakeMultiParam(attrs)), None)
        exp_attrs = {
            'type': 'Permute',
            'op': 'Permute',
            'order': np.array([0, 1, 3, 2]),
            'infer': transpose_infer
        }
        for key in exp_attrs.keys():
            if key == 'order':
                np.testing.assert_equal(res[key], exp_attrs[key])
            else:
                self.assertEqual(res[key], exp_attrs[key])
