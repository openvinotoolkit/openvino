# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.mxnet.conv_ext import DeconvFrontExtractor
from unit_tests.utils.extractors import PB


class TestDeconvShapesParsing(unittest.TestCase):
    def test_conv_ext_ideal_numbers(self):
        params = {'attrs': {
            "kernel": "(4, 4)",
            "no_bias": "True",
            "num_filter": "21",
            "num_group": "14",
            "pad": "(4, 4)",
            "stride": "(2, 2)",
            "dilate": "(3, 3)",
            "workspace": "1536"
        }}
        node = PB({'symbol_dict': params})
        DeconvFrontExtractor.extract(node)
        exp_res = {
            'op': 'Deconvolution',
            'pad': np.array([[0, 0], [0, 0], [4, 4], [4, 4]]),
            'pad_spatial_shape': np.array([[4, 4], [4, 4]]),
            'stride': np.array([1, 1, 2, 2]),
            'kernel_spatial': np.array([4, 4]),
            'dilation': np.array([1, 1, 3, 3]),
            'group': 14,
            'output': 21,
            'bias_addable': True,
            'bias_term': False,
        }
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation'):
                np.testing.assert_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])


    def test_conv_ext_no_bias(self):
        params = { 'attrs':{
            "kernel": "(4, 4)",
            "num_filter": "21",
            "num_group": "14",
            "pad": "(4, 4)",
            "stride": "(2, 2)",
            "dilate": "(3, 3)",
            "workspace": "1536"
        }}
        node = PB({'symbol_dict': params})
        DeconvFrontExtractor.extract(node)
        exp_res = {
            'op': 'Deconvolution',
            'pad': np.array([[0, 0], [0, 0], [4, 4], [4, 4]]),
            'pad_spatial_shape': np.array([[4, 4], [4, 4]]),
            'stride': np.array([1, 1, 2, 2]),
            'kernel_spatial': np.array([4, 4]),
            'dilation': np.array([1, 1, 3, 3]),
            'group': 14,
            'output': 21,
            'bias_addable': True,
            'bias_term': False,
        }
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation'):
                np.testing.assert_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])


    def test_conv_ext_with_bias(self):
        params = { 'attrs':{
            "kernel": "(4, 4)",
            "no_bias": "False",
            "num_filter": "21",
            "num_group": "14",
            "pad": "(4, 4)",
            "stride": "(2, 2)",
            "dilate": "(3, 3)",
            "workspace": "1536"
        }}
        node = PB({'symbol_dict': params})
        DeconvFrontExtractor.extract(node)
        exp_res = {
            'op': 'Deconvolution',
            'pad': np.array([[0, 0], [0, 0], [4, 4], [4, 4]]),
            'pad_spatial_shape': np.array([[4, 4], [4, 4]]),
            'stride': np.array([1, 1, 2, 2]),
            'kernel_spatial': np.array([4, 4]),
            'dilation': np.array([1, 1, 3, 3]),
            'group': 14,
            'output': 21,
            'bias_addable': True,
            'bias_term': True,
        }
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation'):
                np.testing.assert_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])


    def test_deconv_ext_target_shape(self):
        params = {'attrs': {
            "kernel": "(4, 4)",
            "no_bias": "True",
            "num_filter": "21",
            "num_group": "14",
            "pad": "(4, 4)",
            "stride": "(2, 2)",
            "dilate": "(3, 3)",
            "workspace": "1536",
            "target_shape": "(120, 120)"
        }}
        node = PB({'symbol_dict': params})
        DeconvFrontExtractor.extract(node)
        exp_res = {
            'op': 'Deconvolution',
            'pad': np.array([[0, 0], [0, 0], [4, 4], [4, 4]]),
            'pad_spatial_shape': np.array([[4, 4], [4, 4]]),
            'stride': np.array([1, 1, 2, 2]),
            'kernel_spatial': np.array([4, 4]),
            'dilation': np.array([1, 1, 3, 3]),
            'group': 14,
            'output': 21,
            'bias_addable': True,
            'bias_term': False,
            'output_spatial_shape': np.array([120, 120]),
        }
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation', 'output_spatial_shape'):
                np.testing.assert_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])

    def test_deconv_ext_output_pad(self):
        params = {'attrs': {
            "kernel": "(4, 4)",
            "no_bias": "True",
            "num_filter": "21",
            "num_group": "14",
            "pad": "(4, 4)",
            "stride": "(2, 2)",
            "dilate": "(3, 3)",
            "workspace": "1536",
            "adj": "(1, 1)"
        }}
        node = PB({'symbol_dict': params})
        DeconvFrontExtractor.extract(node)
        exp_res = {
            'op': 'Deconvolution',
            'pad': np.array([[0, 0], [0, 0], [4, 4], [4, 4]]),
            'pad_spatial_shape': np.array([[4, 4], [4, 4]]),
            'stride': np.array([1, 1, 2, 2]),
            'kernel_spatial': np.array([4, 4]),
            'dilation': np.array([1, 1, 3, 3]),
            'group': 14,
            'output': 21,
            'bias_addable': True,
            'bias_term': False,
            'output_padding': np.array([0, 0, 1, 1]),
        }
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation', 'output_spatial_shape', 'output_padding'):
                np.testing.assert_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])

    def test_deconv_ext_target_shape_with_output_pad(self):
        params = {'attrs': {
            "kernel": "(4, 4)",
            "no_bias": "True",
            "num_filter": "21",
            "num_group": "14",
            "pad": "(4, 4)",
            "stride": "(2, 2)",
            "dilate": "(3, 3)",
            "workspace": "1536",
            "target_shape": "(120, 120)",
            "adj": "(1, 1)"
        }}
        node = PB({'symbol_dict': params})
        DeconvFrontExtractor.extract(node)
        exp_res = {
            'op': 'Deconvolution',
            'pad': np.array([[0, 0], [0, 0], [4, 4], [4, 4]]),
            'pad_spatial_shape': np.array([[4, 4], [4, 4]]),
            'stride': np.array([1, 1, 2, 2]),
            'kernel_spatial': np.array([4, 4]),
            'dilation': np.array([1, 1, 3, 3]),
            'group': 14,
            'output': 21,
            'bias_addable': True,
            'bias_term': False,
            'output_spatial_shape': np.array([120, 120]),
        }
        for key in exp_res.keys():
            if key in ('pad', 'pad_spatial_shape', 'stride', 'kernel_spatial', 'dilation', 'output_spatial_shape'):
                np.testing.assert_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])
