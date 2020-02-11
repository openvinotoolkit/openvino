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

from extensions.front.tf.concat_ext import ConcatFrontExtractor
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


class ConcatExtractorTest(BaseExtractorsTestingClass):
    def test_concat(self):
        node = PB({'pb': PB({'attr': {'N': PB({'i': 4})}})})
        self.expected = {
            'N': 4,
            'simple_concat': True,
            'type': 'Concat',
            'op': 'Concat',
            'kind': 'op',
            'axis': 1
        }
        ConcatFrontExtractor.extract(node)
        self.res = node
        self.compare()
