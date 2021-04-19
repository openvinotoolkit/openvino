# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from extensions.front.caffe.normalize_ext import NormalizeFrontExtractor
from extensions.ops.normalize import NormalizeOp
from mo.ops.op import Op
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode


class FakeNormalizeProtoLayer:
    def __init__(self, val):
        self.norm_param = val


class TestNormalizeExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Normalize'] = NormalizeOp

    def test_normalize_no_pb_no_ml(self):
        self.assertRaises(AttributeError, NormalizeFrontExtractor.extract, None)

    @patch('extensions.front.caffe.normalize_ext.collect_attributes')
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
