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

from extensions.front.mxnet.slice_channel_ext import SliceChannelFrontExtractor
from mo.utils.unittest.extractors import PB


class TestSliceChannelParsing(unittest.TestCase):
    def test_parse_values(self):
        params = {'attrs': {
            "num_outputs": "2",
            'axis': "2",
        }}

        node = PB({'symbol_dict': params})
        SliceChannelFrontExtractor.extract(node)
        exp_res = {
            'op': 'Split',
            'axis': 2,
            'num_split': 2,
        }

        for key in exp_res.keys():
            self.assertEqual(node[key], exp_res[key])

    def test_parse_dafault_values(self):
        params = {'attrs': {
            "num_outputs": "2",
        }}

        node = PB({'symbol_dict': params})
        SliceChannelFrontExtractor.extract(node)
        exp_res = {
            'op': 'Split',
            'axis': 1,
            'num_split': 2,
        }

        for key in exp_res.keys():
            self.assertEqual(node[key], exp_res[key])
