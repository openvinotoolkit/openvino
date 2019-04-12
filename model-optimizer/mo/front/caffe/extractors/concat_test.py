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

from mo.front.caffe.extractors.concat import concat_ext
from mo.front.common.partial_infer.concat import concat_infer
from mo.utils.unittest.extractors import FakeParam


class FakeProtoLayer:
    def __init__(self, axis):
        self.concat_param = FakeParam('axis', axis)


class TestConcat(unittest.TestCase):
    def test_concat(self):
        res = concat_ext(FakeProtoLayer(10), None)
        exp_res = {
            'axis': 10,
            'infer': concat_infer,
            'type': 'Concat'
        }
        self.assertEqual(res, exp_res)
