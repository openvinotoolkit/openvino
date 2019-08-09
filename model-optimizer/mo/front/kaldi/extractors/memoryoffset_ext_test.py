"""
 Copyright (c) 2019 Intel Corporation

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
from mo.front.kaldi.extractors.memoryoffset_ext import MemoryOffsetFrontExtractor
from mo.ops.memoryoffset import MemoryOffset
from mo.ops.op import Op


class MemoryOffsetFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['memoryoffset'] = MemoryOffset

    @classmethod
    def create_pb_for_test_node(cls):
        pb = {'pair_name': 'my_pair',
              't': -5,
              'has_default': False
              }
        cls.test_node['parameters'] = pb

    def test_extract(self):
        MemoryOffsetFrontExtractor.extract(self.test_node)
        self.assertEqual(self.test_node['pair_name'], 'my_pair')
        self.assertEqual(self.test_node['t'], -5)
