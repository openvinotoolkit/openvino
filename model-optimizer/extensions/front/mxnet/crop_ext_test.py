# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.front.mxnet.crop_ext import CropFrontExtractor
from mo.ops.crop import Crop
from mo.utils.unittest.extractors import PB


class TestCropExt(unittest.TestCase):
    def test_crop_ext(self):
        params = {
            'attrs': {
                'offset': '(5, 5)',
                'num_args': 2
            }
        }
        node = PB({'symbol_dict': params})
        CropFrontExtractor.extract(node)

        exp_res = {
            'axis': 2,
            'offset': [5, 5],
            'dim': None,
            'infer': Crop.infer,
            'type': 'Crop'
        }
        for key in exp_res.keys():
            self.assertEqual(node[key], exp_res[key])
