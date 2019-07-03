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

from extensions.front.tf.pad_ext import PadFrontExtractor
from mo.utils.unittest.extractors import PB


class TestPad(unittest.TestCase):
    def test_no_pads(self):
        node = PB({})
        PadFrontExtractor.extract(node)
        self.assertTrue(not 'pads' in node or node['pads'] is None)
