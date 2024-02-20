# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.ctcgreedydecoder_ext import CTCGreedyDecoderFrontExtractor
from openvino.tools.mo.ops.ctc_greedy_decoder import CTCGreedyDecoderOp
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakeCTCGreedyDecoderProtoLayer:
    def __init__(self, val):
        self.ctc_decoder_param = val


class TestCTCGreedyDecoderExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['CTCGreedyDecoder'] = CTCGreedyDecoderOp

    def test_ctcgreedydecoder_no_pb_no_ml(self):
        self.assertRaises(AttributeError, CTCGreedyDecoderFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.ctcgreedydecoder_ext.merge_attrs')
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
            'infer': CTCGreedyDecoderOp.infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
