# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from extensions.front.caffe.reorgyolo_ext import ReorgYoloFrontExtractor
from extensions.ops.reorgyolo import ReorgYoloOp
from mo.ops.op import Op
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode


class FakeReorgYoloProtoLayer:
    def __init__(self, val):
        self.reorg_yolo_param = val


class TestReorgYoloExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['ReorgYolo'] = ReorgYoloOp

    def test_elu_no_pb_no_ml(self):
        self.assertRaises(AttributeError, ReorgYoloFrontExtractor.extract, None)

    @patch('extensions.front.caffe.reorgyolo_ext.merge_attrs')
    def test_elu_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'stride': 2
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakeReorgYoloProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        ReorgYoloFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "ReorgYolo",
            'stride': 2,
            'infer': ReorgYoloOp.reorgyolo_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
