# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.regionyolo_ext import RegionYoloFrontExtractor
from openvino.tools.mo.ops.regionyolo import RegionYoloOp
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakeRegionYoloProtoLayer:
    def __init__(self, val, val_f):
        self.region_yolo_param = val
        self.flatten_param = val_f


class TestReorgYoloExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['RegionYolo'] = RegionYoloOp

    def test_reogyolo_no_pb_no_ml(self):
        self.assertRaises(AttributeError, RegionYoloFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.regionyolo_ext.merge_attrs')
    def test_reogyolo_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'coords': 4,
            'classes': 20,
            'num': 5,
            'do_softmax': 1,
            'anchors': 5,
            'mask': 5,
        }
        params_flatten = {
            'axis': 1,
            'end_axis': -1
        }
        merge_attrs_mock.return_value = {
            **params,
            **params_flatten
        }

        fake_pl = FakeRegionYoloProtoLayer(FakeMultiParam(params), FakeMultiParam(params_flatten))
        fake_node = FakeNode(fake_pl, None)

        RegionYoloFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "RegionYolo",
            'coords': 4,
            'classes': 20,
            'num': 5,
            'axis': 1,
            'end_axis': -1,
            'do_softmax': 1,
            'anchors': 5,
            'mask': 5,
            'infer': RegionYoloOp.regionyolo_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
