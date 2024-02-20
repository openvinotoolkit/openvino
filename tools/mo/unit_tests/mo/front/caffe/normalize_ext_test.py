# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.normalize_ext import NormalizeFrontExtractor
from openvino.tools.mo.ops.normalize import NormalizeOp
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakeNormalizeProtoLayer:
    def __init__(self, val):
        self.norm_param = val


class TestNormalizeExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Normalize'] = NormalizeOp

    def test_normalize_no_pb_no_ml(self):
        self.assertRaises(AttributeError, NormalizeFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.normalize_ext.collect_attributes')
    def test_normalize_ext_ideal_numbers(self, collect_attributes_mock):
        params = {
            'across_spatial': 1,
            'channel_shared': 0,
            'eps': 0.00001
        }
        collect_attributes_mock.return_value = {
            **params
        }

        fake_pl = FakeNormalizeProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        NormalizeFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Normalize",
            'across_spatial': 1,
            'channel_shared': 0,
            'eps': 0.00001,
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
