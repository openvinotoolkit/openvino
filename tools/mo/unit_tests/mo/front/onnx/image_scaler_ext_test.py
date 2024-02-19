# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import onnx

from openvino.tools.mo.front.onnx.image_scaler_ext import ImageScalerFrontExtractor
from unit_tests.utils.extractors import PB


class TestImageScalerONNXExt(unittest.TestCase):
    @staticmethod
    def _create_image_scaler_node():
        pb = onnx.helper.make_node(
            'ImageScaler',
            inputs=['a'],
            outputs=['b'],
            scale=1.0,
            bias=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        node = PB({'pb': pb, 'graph': PB({'graph': {'layout': 'NCHW'}})})
        return node

    def test_image_scaler_ext(self):
        node = self._create_image_scaler_node()
        ImageScalerFrontExtractor.extract(node)

        exp_res = {
            'scale': 1.0,
            'bias': [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]], [[7.0]], [[8.0]]],
        }

        for key in exp_res.keys():
            if type(node[key]) in [list, np.ndarray]:
                self.assertTrue(np.array_equal(np.array(node[key]), np.array(exp_res[key])))
            else:
                self.assertEqual(node[key], exp_res[key])
