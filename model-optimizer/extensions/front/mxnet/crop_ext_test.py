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
