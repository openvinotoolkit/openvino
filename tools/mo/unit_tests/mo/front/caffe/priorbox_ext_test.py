# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np

from openvino.tools.mo.front.caffe.priorbox_ext import PriorBoxFrontExtractor
from openvino.tools.mo.ops.priorbox import PriorBoxOp
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam, FakeParam
from unit_tests.utils.graph import FakeNode


class FakeMultiParamListFields(FakeMultiParam):
    def __init__(self, val):
        super().__init__(val)

    def ListFields(self):
        keys = []
        for k in self.dict_values.keys():
            keys.append([FakeParam('name', k)])
        return keys


class FakePriorBoxProtoLayer:
    def __init__(self, val):
        self.prior_box_param = val


class TestPriorBoxExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['PriorBox'] = PriorBoxOp

    def test_priorbox_no_pb_no_ml(self):
        self.assertRaises(AttributeError, PriorBoxFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.priorbox_ext.merge_attrs')
    def test_priorbox_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'clip': False,
            'flip': True,
            'min_size': np.array([]),
            'max_size': np.array([]),
            'aspect_ratio': np.array([2, 3]),
            'variance': np.array(['0.2', '0.3', '0.2', '0.3']),
            'img_size': '300',
            'img_h': '0',
            'img_w': '0',
            'step': '0,5',
            'step_h': '0',
            'step_w': '0',
            'offset': '0.6'
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakePriorBoxProtoLayer(FakeMultiParamListFields(params))
        fake_node = FakeNode(fake_pl, None)

        PriorBoxFrontExtractor.extract(fake_node)

        exp_res = {
            'op': 'PriorBox',
            'type': 'PriorBox',
            'clip': 0,
            'variance': np.array(['0.2', '0.3', '0.2', '0.3']),
            'img_size': '300',
            'img_h': '0',
            'img_w': '0',
            'step': '0,5',
            'step_h': '0',
            'step_w': '0',
            'offset': '0.6'
        }

        for key in exp_res.keys():
            if key in ['width', 'height', 'variance']:
                np.testing.assert_equal(fake_node[key], exp_res[key])
            else:
                self.assertEqual(fake_node[key], exp_res[key])

    @patch('openvino.tools.mo.front.caffe.priorbox_ext.merge_attrs')
    def test_priorbox_ext_ideal_numbers_density(self, merge_attrs_mock):
        params = {
            'clip': False,
            'flip': True,
            'min_size': np.array([]),
            'max_size': np.array([]),
            'aspect_ratio': np.array([2, 3]),
            'variance': np.array(['0.2', '0.3', '0.2', '0.3']),
            'img_size': '300',
            'img_h': '0',
            'img_w': '0',
            'step': '0,5',
            'step_h': '0',
            'step_w': '0',
            'offset': '0.6',
            'fixed_size': np.array(['1', '32']),
            'fixed_ratio': np.array(['0.2', '0.5']),
            'density': np.array(['0.3', '0.6'])
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakePriorBoxProtoLayer(FakeMultiParamListFields(params))
        fake_node = FakeNode(fake_pl, None)

        PriorBoxFrontExtractor.extract(fake_node)

        exp_res = {
            'op': 'PriorBox',
            'type': 'PriorBox',
            'clip': 0,
            'variance': np.array(['0.2', '0.3', '0.2', '0.3']),
            'img_size': '300',
            'img_h': '0',
            'img_w': '0',
            'step': '0,5',
            'step_h': '0',
            'step_w': '0',
            'offset': '0.6',
            'fixed_size': np.array(['1', '32']),
            'fixed_ratio': np.array(['0.2', '0.5']),
            'density': np.array(['0.3', '0.6'])
        }

        for key in exp_res.keys():
            if key in ['width', 'height', 'variance', 'fixed_size', 'fixed_ratio', 'density']:
                np.testing.assert_equal(fake_node[key], exp_res[key])
            else:
                self.assertEqual(fake_node[key], exp_res[key])
