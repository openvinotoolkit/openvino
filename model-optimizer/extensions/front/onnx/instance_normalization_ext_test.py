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

from extensions.front.onnx.instance_normalization_ext import InstanceNormalizationExtractor
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


class TestInstanceNormalization(BaseExtractorsTestingClass):
    @staticmethod
    def _create_node():
        pb = onnx.helper.make_node(
            'InstanceNormalization',
            inputs=['a'],
            outputs=['b'],
            epsilon=0.5,
        )
        node = PB({'pb': pb})
        return node

    def test_image_scaler_ext(self):
        node = self._create_node()
        InstanceNormalizationExtractor.extract(node)
        self.res = node

        self.expected = {
            'epsilon': 0.5,
        }

        self.compare()
