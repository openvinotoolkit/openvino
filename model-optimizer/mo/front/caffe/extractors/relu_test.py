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

from mo.front.caffe.extractors.relu import relu_ext
from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.utils.unittest.extractors import FakeParam, FakeMultiParam


class TestReLU(unittest.TestCase):
    def test_relu_ext(self):
        params = {
            'negative_slope': 0.1,
        }

        res = relu_ext(FakeParam('relu_param', FakeMultiParam(params)), None)
        exp_res = {
            'negative_slope': 0.1,
            'infer': copy_shape_infer,
        }
        exp_res.update(layout_attrs())
        for i in exp_res.keys():
            if i == 'negative_slope':
                self.assertEqual(res[i], exp_res[i])
            else:
                np.testing.assert_array_equal(res[i], exp_res[i])
