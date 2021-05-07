# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from extensions.front.caffe.crop_ext import CropFrontExtractor
from mo.front.common.partial_infer.crop import crop_infer
from mo.ops.crop import Crop
from mo.ops.op import Op
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode


class FakeCropProtoLayer:
    def __init__(self, val):
        self.crop_param = val


class TestCropExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Crop'] = Crop

    def test_da_no_pb_no_ml(self):
        self.assertRaises(AttributeError, CropFrontExtractor.extract, None)

    @patch('mo.front.caffe.collect_attributes')
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
            'type': 'Crop',
            'axis': 0,
            'offset': 0,
            'dim': None,  # set in infer
            'infer': crop_infer
        }

        for key in exp_res.keys():
            self.assertEqual(exp_res[key], fake_node[key])
