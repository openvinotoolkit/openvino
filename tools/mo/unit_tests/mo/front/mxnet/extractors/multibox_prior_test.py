# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.mxnet.extractors.multibox_prior import multi_box_prior_ext
from openvino.tools.mo.front.mxnet.extractors.utils import AttrDictionary


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
            'step': 0.02666666666666667,
            'offset': 0.5,
            'variance': '0.100000,0.100000,0.200000,0.200000',
            'flip': 0,
            'clip': 0,
            'min_size': [0.1, 0.141],
            'max_size': '',
            'aspect_ratio': [1, 2, 0.5],
        }

        for key in exp_attrs.keys():
            if key in ['aspect_ratio', 'variance']:
                np.testing.assert_equal(res[key], exp_attrs[key])
            else:
                self.assertEqual(res[key], exp_attrs[key])
