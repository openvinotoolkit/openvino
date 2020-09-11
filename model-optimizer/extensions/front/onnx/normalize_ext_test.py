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

import onnx

from extensions.front.onnx.normalize_ext import NormalizeFrontExtractor
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


class TestNormalize(BaseExtractorsTestingClass):
    @staticmethod
    def _create_node(across_spatial=None, channel_shared=None, eps=None):
        if across_spatial is None:
            across_spatial = 0
        if channel_shared is None:
            channel_shared = 0
        if eps is None:
            eps = 0.1
        pb = onnx.helper.make_node(
            'Normalize',
            across_spatial=across_spatial,
            channel_shared=channel_shared,
            eps=eps,
            inputs=['a'],
            outputs=['b']
        )
        node = PB({'pb': pb})
        return node

    def test_ok(self):
        node = self._create_node()
        NormalizeFrontExtractor.extract(node)
        self.res = node

        self.expected = {
            'across_spatial': False,
            'channel_shared': False,
            'eps': 0.1
        }

        self.compare()
