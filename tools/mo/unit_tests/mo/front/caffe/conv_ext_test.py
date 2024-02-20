# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np

from openvino.tools.mo.front.caffe.conv_ext import ConvFrontExtractor, DeconvFrontExtractor, conv_create_attrs, conv_set_params
from openvino.tools.mo.front.caffe.extractors.utils import get_list_from_container
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.extractors import PB, FakeParam, FakeMultiParam


class FakeConvProtoLayer:
    def __init__(self, val):
        self.convolution_param = val
        self.bottom = [0]


class TestConvShapesParsing(unittest.TestCase):
    def test_conv_no_pb_no_ml(self):
        node = PB({'pb': None})
        self.assertRaises(Error, ConvFrontExtractor.extract, node)

    @patch('openvino.tools.mo.front.caffe.conv_ext.weights_biases')
    @patch('openvino.tools.mo.front.caffe.conv_ext.layout_attrs')
    def test_conv_ext_ideal_numbers(self, weights_biases_mock, layout_attrs_mock):
        weights_biases_mock.return_value = {}
        layout_attrs_mock.return_value = {}
        params = {
            'pad': 10,
            'kernel_size': 11,
            'stride': 12,
            'dilation': 13,
            'group': 14,
            'num_output': 15,
            'bias_term': True
        }
        node = PB({'pb': FakeConvProtoLayer(FakeMultiParam(params))})
        ConvFrontExtractor.extract(node)
        res = node
        exp_res = {
            'op': 'Conv2D',
            'pad': np.array([[0, 0], [0, 0], [10, 10], [10, 10]]),
            'pad_spatial_shape': np.array([[10, 10], [10, 10]]),
            'stride': np.array([1, 1, 12, 12]),
            'kernel_spatial': np.array([11, 11]),
            'dilation': np.array([1, 1, 13, 13]),
            'group': 14,
            'bias_addable': True,
            'bias_term': True,
        }
        self.assertTrue(weights_biases_mock.called)
        self.assertTrue(layout_attrs_mock.called)
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation'):
                np.testing.assert_equal(res[key], exp_res[key])
            else:
                self.assertEqual(res[key], exp_res[key])

    @patch('openvino.tools.mo.front.caffe.conv_ext.weights_biases')
    @patch('openvino.tools.mo.front.caffe.conv_ext.layout_attrs')
    def test_conv_ext_empty_numbers(self, weights_biases_mock, layout_attrs_mock):
        weights_biases_mock.return_value = {}
        layout_attrs_mock.return_value = {}
        params = {
            'pad': None,
            'kernel_size': None,
            'stride': None,
            'dilation': None,
            'group': 14,
            'num_output': 15,
            'bias_term': True,
            'pad_w': 3,
            'pad_h': 4,
            'kernel_w': 5,
            'kernel_h': 6,
            'stride_h': 3,
            'stride_w': 2,
        }
        node = PB({'pb': FakeConvProtoLayer(FakeMultiParam(params))})
        ConvFrontExtractor.extract(node)
        res = node
        exp_res = {
            'op': 'Conv2D',
            'pad': np.array([[0, 0], [0, 0], [4, 4], [3, 3]]),
            'pad_spatial_shape': np.array([[4, 4], [3, 3]]),
            'stride': np.array([1, 1, 3, 2]),
            'kernel_spatial': np.array([6, 5]),
            'dilation': np.array([1, 1, 1, 1]),
            'group': 14,
            'bias_addable': True,
            'bias_term': True,
        }
        self.assertTrue(weights_biases_mock.called)
        self.assertTrue(layout_attrs_mock.called)
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation'):
                np.testing.assert_equal(res[key], exp_res[key])
            else:
                self.assertEqual(res[key], exp_res[key])

    def test_attrs(self):
        params = {
            'type_str': 'Conv2D',
            'padding': [10, 10],
            'stride': [12, 12],
            'kernel': [11, 11],
            'dilate': [13, 13],
            'group': 14,
            'output': 13,
            'bias_term': True
        }

        res = conv_create_attrs(params)

        exp_res = {
            'pad': np.array([[0, 0], [0, 0], [10, 10], [10, 10]]),
            'pad_spatial_shape': np.array([[10, 10], [10, 10]]),
            'stride': np.array([1, 1, 12, 12]),
            'kernel_spatial': np.array([11, 11]),
            'dilation': np.array([1, 1, 13, 13]),
            'group': 14,
            'bias_addable': True,
            'bias_term': True,
            'output_spatial_shape': None,
            'output_shape': None,
            'output': 13,
        }
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation'):
                np.testing.assert_equal(res[key], exp_res[key])
            else:
                self.assertEqual(res[key], exp_res[key])

    def test_get_list_from_container_no_existing_param(self):
        res = get_list_from_container(FakeParam("p", "1"), 'prop', int)
        self.assertEqual(res, [])

    def test_get_list_from_container_no_param(self):
        res = get_list_from_container(None, 'prop', int)
        self.assertEqual(res, [])

    def test_get_list_from_container_simple_type_match(self):
        res = get_list_from_container(FakeParam('prop', 10), 'prop', int)
        self.assertEqual(res, [10])

    def test_get_list_from_container_list_match(self):
        res = get_list_from_container(FakeParam('prop', [10, 11]), 'prop', int)
        self.assertEqual(res, [10, 11])

    def test_get_list_from_container_list_match_empty(self):
        res = get_list_from_container(FakeParam('prop', []), 'prop', int)
        self.assertEqual(res, [])

    def test_params_creation(self):
        params = {
            'pad': None,
            'kernel_size': None,
            'stride': None,
            'dilation': None,
            'group': 14,
            'num_output': 15,
            'bias_term': True,
            'pad_w': 3,
            'pad_h': 4,
            'kernel_w': 5,
            'kernel_h': 6,
            'stride_h': 3,
            'stride_w': 2,
        }
        exp_res = {
            'padding': [3, 4],
            'stride': [2, 3],
            'kernel': [5, 6],
            'dilate': [1, 1],
            'group': 14,
            'output': 15
        }
        res = conv_set_params(FakeConvProtoLayer(FakeMultiParam(params)).convolution_param, 'Conv2D')

        for key in exp_res.keys():
            if key in ('padding', 'stride', 'stride', 'kernel', 'dilate'):
                np.testing.assert_equal(res[key], exp_res[key])
            else:
                self.assertEqual(res[key], exp_res[key])


class TestDeconvShapesParsing(unittest.TestCase):
    def test_deconv_no_pb_no_ml(self):
        node = PB({'pb': None})
        self.assertRaises(Error, DeconvFrontExtractor.extract, node)

    @patch('openvino.tools.mo.front.caffe.conv_ext.weights_biases')
    @patch('openvino.tools.mo.front.caffe.conv_ext.layout_attrs')
    def test_conv_ext_ideal_numbers(self, weights_biases_mock, layout_attrs_mock):
        weights_biases_mock.return_value = {}
        layout_attrs_mock.return_value = {}
        params = {
            'pad': 10,
            'kernel_size': 11,
            'stride': 12,
            'dilation': 13,
            'group': 14,
            'num_output': 15,
            'bias_term': True
        }
        node = PB({'pb': FakeConvProtoLayer(FakeMultiParam(params))})
        res = DeconvFrontExtractor.extract(node)
        res = node
        exp_res = {
            'op': 'Deconv2D',
            'pad': np.array([[0, 0], [0, 0], [10, 10], [10, 10]]),
            'pad_spatial_shape': np.array([[10, 10], [10, 10]]),
            'stride': np.array([1, 1, 12, 12]),
            'kernel_spatial': np.array([11, 11]),
            'dilation': np.array([1, 1, 13, 13]),
            'group': 14,
            'bias_addable': True,
        }
        self.assertTrue(weights_biases_mock.called)
        self.assertTrue(layout_attrs_mock.called)
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation'):
                np.testing.assert_equal(res[key], exp_res[key])
            else:
                self.assertEqual(res[key], exp_res[key])

    @patch('openvino.tools.mo.front.caffe.conv_ext.weights_biases')
    @patch('openvino.tools.mo.front.caffe.conv_ext.layout_attrs')
    def test_conv_ext_false_bias_term(self, weights_biases_mock, layout_attrs_mock):
        weights_biases_mock.return_value = {}
        layout_attrs_mock.return_value = {}
        params = {
            'pad': 10,
            'kernel_size': 11,
            'stride': 12,
            'dilation': 13,
            'group': 14,
            'num_output': 15,
            'bias_term': False
        }
        node = PB({'pb': FakeConvProtoLayer(FakeMultiParam(params))})
        res = DeconvFrontExtractor.extract(node)
        res = node
        exp_res = {
            'op': 'Deconv2D',
            'pad': np.array([[0, 0], [0, 0], [10, 10], [10, 10]]),
            'pad_spatial_shape': np.array([[10, 10], [10, 10]]),
            'stride': np.array([1, 1, 12, 12]),
            'kernel_spatial': np.array([11, 11]),
            'dilation': np.array([1, 1, 13, 13]),
            'group': 14,
            'bias_addable': True,
            'bias_term': False,
        }
        self.assertTrue(weights_biases_mock.called)
        self.assertTrue(layout_attrs_mock.called)
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation', 'bias_term'):
                np.testing.assert_equal(res[key], exp_res[key])
            else:
                self.assertEqual(res[key], exp_res[key])

    @patch('openvino.tools.mo.front.caffe.conv_ext.weights_biases')
    @patch('openvino.tools.mo.front.caffe.conv_ext.layout_attrs')
    def test_conv_ext_empty_numbers(self, weights_biases_mock, layout_attrs_mock):
        weights_biases_mock.return_value = {}
        layout_attrs_mock.return_value = {}
        params = {
            'pad': None,
            'kernel_size': None,
            'stride': None,
            'dilation': None,
            'group': 14,
            'num_output': 15,
            'bias_term': True,
            'pad_w': 3,
            'pad_h': 4,
            'kernel_w': 5,
            'kernel_h': 6,
            'stride_h': 3,
            'stride_w': 2,
        }
        node = PB({'pb': FakeConvProtoLayer(FakeMultiParam(params))})
        res = DeconvFrontExtractor.extract(node)
        res = node
        exp_res = {
            'op': 'Deconv2D',
            'pad': np.array([[0, 0], [0, 0], [4, 4], [3, 3]]),
            'pad_spatial_shape': np.array([[4, 4], [3, 3]]),
            'stride': np.array([1, 1, 3, 2]),
            'kernel_spatial': np.array([6, 5]),
            'dilation': np.array([1, 1, 1, 1]),
            'group': 14,
            'bias_addable': True,
        }
        self.assertTrue(weights_biases_mock.called)
        self.assertTrue(layout_attrs_mock.called)
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation'):
                np.testing.assert_equal(res[key], exp_res[key])
            else:
                self.assertEqual(res[key], exp_res[key])

    def test_attrs(self):
        params = {
            'type_str': 'Deconv2D',
            'padding': [10, 10],
            'stride': [12, 12],
            'kernel': [11, 11],
            'dilate': [13, 13],
            'group': 14,
            'output': 13,
            'bias_term': True
        }
        res = conv_create_attrs(params)

        exp_res = {
            'pad': np.array([[0, 0], [0, 0], [10, 10], [10, 10]]),
            'pad_spatial_shape': np.array([[10, 10], [10, 10]]),
            'stride': np.array([1, 1, 12, 12]),
            'kernel_spatial': np.array([11, 11]),
            'dilation': np.array([1, 1, 13, 13]),
            'group': 14,
            'bias_addable': True,
            'output_spatial_shape': None,
            'output_shape': None,
            'output': 13,
        }
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation'):
                np.testing.assert_equal(res[key], exp_res[key])
            else:
                self.assertEqual(res[key], exp_res[key])
