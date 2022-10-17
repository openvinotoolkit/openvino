# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import onnx
from generator import generator, generate

from openvino.tools.mo.front.onnx.squeeze_ext import SqueezeFrontExtractor
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.squeeze import Squeeze
from unit_tests.utils.extractors import PB


@generator
class TestSqueezeONNXExt(unittest.TestCase):
    @staticmethod
    def _create_squeeze_node(axes):
        if axes is None:
            pb = onnx.helper.make_node(
                'Squeeze',
                inputs=['x'],
                outputs=['y'],
            )
        else:
            pb = onnx.helper.make_node(
                'Squeeze',
                inputs=['x'],
                outputs=['y'],
                axes=axes,
            )

        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Squeeze'] = Squeeze

    @generate(*[[0, 1, 2, 3], [1], None])
    def test_squeeze_ext(self, axes):
        node = self._create_squeeze_node(axes)
        SqueezeFrontExtractor.extract(node)

        exp_res = {
            'type': 'Squeeze',
            'squeeze_dims': np.array(axes),
        }

        assert node['type'] == exp_res['type']
        assert np.array_equal(node['squeeze_dims'], exp_res['squeeze_dims'])
