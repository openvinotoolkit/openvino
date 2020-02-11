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

from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.front.kaldi.extractors.slice_ext import SliceFrontExtractor
from mo.ops.op import Op
from mo.ops.slice import Slice
from mo.utils.unittest.extractors import FakeMultiParam


class SliceFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Slice'] = Slice
        cls.slice_params = {
            'slice_point': [99, 1320],
            'axis': 1
        }
        cls.test_node['pb'] = FakeMultiParam(cls.slice_params)

    def test_assertion_no_pb(self):
        self.assertRaises(AttributeError, SliceFrontExtractor.extract, None)
