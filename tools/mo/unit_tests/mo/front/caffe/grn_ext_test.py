# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.grn_ext import GRNFrontExtractor
from openvino.tools.mo.ops.grn import GRNOp
from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakeGRNProtoLayer:
    def __init__(self, val):
        self.grn_param = val


class TestGRNExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['GRN'] = GRNOp

    def test_grn_no_pb_no_ml(self):
        self.assertRaises(AttributeError, GRNFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.grn_ext.merge_attrs')
    def test_grn_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'bias': 0.7
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakeGRNProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        GRNFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "GRN",
            'bias': 0.7,
            'infer': copy_shape_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
