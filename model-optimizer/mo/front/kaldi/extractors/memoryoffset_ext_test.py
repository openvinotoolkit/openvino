# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
