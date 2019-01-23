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

from mo.front.mxnet.extractors.multibox_prior import multi_box_prior_ext
from mo.front.mxnet.extractors.utils import AttrDictionary


class TestMultiBoxPrior_Parsing(unittest.TestCase):
    def test_multi_box_prior_check_attrs(self):
        attrs = {
            'ratios': '(1,2,0.5)',
            'steps': '(0.02666666666666667, 0.02666666666666667)',
            'clip': 'False',
            'sizes': '(0.1,0.141)'
        }

        res = multi_box_prior_ext(AttrDictionary(attrs))
        exp_attrs = {
            'type': 'PriorBox',
            'img_size': 0,
            'img_h': 0,
            'img_w': 0,
            'step': 0.02666666666666667,
            'step_h': 0,
            'step_w': 0,
            'offset': 0.5,
            'variance': '0.100000,0.100000,0.200000,0.200000',
            'flip': 0,
            'clip': 0,
            'min_size': (0.1, 0.141),
            'max_size': '',
            'aspect_ratio': [1, 2, 0.5],
        }

        for key in exp_attrs.keys():
            if key in ['aspect_ratio', 'variance']:
                np.testing.assert_equal(res[key], exp_attrs[key])
            else:
                self.assertEqual(res[key], exp_attrs[key])
