# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from extensions.front.caffe.accum_ext import AccumFrontExtractor
from extensions.ops.accum import AccumOp
from mo.ops.op import Op
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode


class FakeAccumProtoLayer:
    def __init__(self, val):
        self.accum_param = val


class TestAccumExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Accum'] = AccumOp

    def test_accum_no_pb_no_ml(self):
        self.assertRaises(AttributeError, AccumFrontExtractor.extract, None)

    @patch('extensions.front.caffe.accum_ext.collect_attributes')
    def test_accum_ext(self, collect_attributes_mock):
        params = {
            'top_height': 200,
            'top_width': 300,
            'size_divisible_by': 3,
            'have_reference': 'False',
        }
        collect_attributes_mock.return_value = {
            **params,
            'have_reference': 0
        }

        fake_pl = FakeAccumProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        AccumFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Accum",
            'top_height': 200,
            'top_width': 300,
            'size_divisible_by': 3,
            'have_reference': 0,
            'infer': AccumOp.accum_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
