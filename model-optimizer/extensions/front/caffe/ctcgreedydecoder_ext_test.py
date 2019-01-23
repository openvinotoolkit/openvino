"""
 Copyright (c) 2018 Intel Corporation

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
from unittest.mock import patch

from extensions.front.caffe.ctcgreedydecoder_ext import CTCGreedyDecoderFrontExtractor
from extensions.ops.ctc_greedy_decoder import CTCGreedyDecoderOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakeCTCGreedyDecoderProtoLayer:
    def __init__(self, val):
        self.ctc_decoder_param = val


class TestCTCGreedyDecoderExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['CTCGreedyDecoder'] = CTCGreedyDecoderOp

    def test_ctcgreedydecoder_no_pb_no_ml(self):
        self.assertRaises(AttributeError, CTCGreedyDecoderFrontExtractor.extract, None)

    @patch('extensions.front.caffe.ctcgreedydecoder_ext.merge_attrs')
    def test_ctcgreedydecoder_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'ctc_merge_repeated': True
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakeCTCGreedyDecoderProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        CTCGreedyDecoderFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "CTCGreedyDecoder",
            'ctc_merge_repeated': 1,
            'infer': CTCGreedyDecoderOp.ctc_greedy_decoder_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
