# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from extensions.front.caffe.correlation_ext import CorrelationFrontExtractor
from extensions.ops.correlation import CorrelationOp
from mo.ops.op import Op
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode


class FakeCorrProtoLayer:
    def __init__(self, val):
        self.correlation_param = val


class TestCorrelationExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Correlation'] = CorrelationOp

    def test_da_no_pb_no_ml(self):
        self.assertRaises(AttributeError, CorrelationFrontExtractor.extract, None)

    @patch('extensions.front.caffe.correlation_ext.merge_attrs')
    def test_resample_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'pad': 20,
            'kernel_size': 1,
            'max_displacement': 20,
            'stride_1': 1,
            'stride_2': 2,
            'single_direction': 0,
            'do_abs': False,
            'correlation_type': 'caffe.CorrelationParameter.MULTIPLY'
        }
        merge_attrs_mock.return_value = {
            **params,
            'test': 54,
            'test2': 'test3'
        }

        fake_pl = FakeCorrProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        CorrelationFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Correlation",
            'pad': 20,
            'kernel_size': 1,
            'max_displacement': 20,
            'stride_1': 1,
            'stride_2': 2,
            'single_direction': 0,
            'do_abs': False,
            'correlation_type': 'caffe.CorrelationParameter.MULTIPLY',
            'infer': CorrelationOp.corr_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
