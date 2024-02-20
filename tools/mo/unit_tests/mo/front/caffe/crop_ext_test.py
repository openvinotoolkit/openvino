# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.crop_ext import CropFrontExtractor
from openvino.tools.mo.front.common.partial_infer.crop import crop_infer
from openvino.tools.mo.ops.crop import Crop
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakeCropProtoLayer:
    def __init__(self, val):
        self.crop_param = val


class TestCropExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Crop'] = Crop

    def test_da_no_pb_no_ml(self):
        self.assertRaises(AttributeError, CropFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.collect_attributes')
    def test_crop_ext(self, collect_attributes_mock):
        params = {
            'axis': 0,
            'offset': 0,
        }
        collect_attributes_mock.return_value = {
            **params,
            'test': 54,
            'test2': 'test3'
        }
        fake_pl = FakeCropProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        CropFrontExtractor.extract(fake_node)

        exp_res = {
            'op': 'Crop',
            'axis': 0,
            'offset': 0,
            'dim': None,  # set in infer
            'infer': crop_infer
        }

        for key in exp_res.keys():
            self.assertEqual(exp_res[key], fake_node[key])
