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

import onnx

from extensions.front.onnx.pad_ext import PadFrontExtractor
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


class TestPad(BaseExtractorsTestingClass):
    @staticmethod
    def _create_node(pads=None, value=None, mode=None):
        if pads is None:
            pads = [1, 2, 3, 4]
        if value is None:
            value = 0.0
        if mode is None:
            mode = 'constant'
        pb = onnx.helper.make_node(
            'Pad',
            pads=pads,
            mode=mode,
            value=value,
            inputs=['a'],
            outputs=['b']
        )
        node = PB({'pb': pb})
        return node

    def test_ok(self):
        node = self._create_node()
        PadFrontExtractor.extract(node)
        self.res = node

        self.expected = {
            'pads': [[1, 3], [2, 4]],
            'mode': 'constant',
            'fill_value': 0
        }

        self.compare()

    def test_reflect(self):
        node = self._create_node(mode='reflect')
        PadFrontExtractor.extract(node)
        self.res = node

        self.expected = {
            'pads': [[1, 3], [2, 4]],
            'mode': 'reflect',
            'fill_value': 0
        }

        self.compare()

    def test_non_zero_fill_value(self):
        node = self._create_node(value=1.0)
        PadFrontExtractor.extract(node)
        self.res = node

        self.expected = {
            'pads': [[1, 3], [2, 4]],
            'mode': 'constant',
            'fill_value': 1.0
        }

        self.compare()
