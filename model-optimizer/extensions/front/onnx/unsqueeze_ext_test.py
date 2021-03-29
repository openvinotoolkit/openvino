# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import onnx
from generator import generator, generate

from extensions.front.onnx.unsqueeze_ext import UnsqueezeFrontExtractor
from mo.ops.op import Op
from mo.ops.unsqueeze import Unsqueeze
from mo.utils.unittest.extractors import PB


@generator
class TestUnsqueezeONNXExt(unittest.TestCase):
    @staticmethod
    def _create_unsqueeze_node(axes):
        pb = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=axes,
        )

        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Unsqueeze'] = Unsqueeze

    @generate(*[[0, 1, 2, 3], [1]])
    def test_unsqueeze_ext(self, axes):
        node = self._create_unsqueeze_node(axes)
        UnsqueezeFrontExtractor.extract(node)

        exp_res = {
            'expand_axis': axes,
        }

        for key in exp_res.keys():
            if type(node[key]) in [list, np.ndarray]:
                self.assertTrue(np.array_equal(np.array(node[key]), np.array(exp_res[key])))
            else:
                self.assertEqual(node[key], exp_res[key])
