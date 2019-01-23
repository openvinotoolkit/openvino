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

from mo.front.common.partial_infer.crop import crop_infer
from mo.front.mxnet.extractors.crop import crop_ext
from mo.front.mxnet.extractors.utils import AttrDictionary


class FakeProtoLayer:
    def __init__(self, val):
        self.crop_param = val


class TestCropExt(unittest.TestCase):
    def test_crop_ext(self):
        params = {
            'offset': '(5, 5)',
            'num_args': 2
        }
        res = crop_ext(AttrDictionary(params))
        exp_res = {
            'axis': 2,
            'offset': [5, 5],
            'dim': None,
            'infer': crop_infer,
            'type': 'Crop'
        }
        for key in exp_res.keys():
            self.assertEqual(res[key], exp_res[key])
