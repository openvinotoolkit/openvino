# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import onnx

from openvino.tools.mo.front.onnx.priorbox_clustered_ext import PriorBoxClusteredFrontExtractor
from openvino.tools.mo.ops.priorbox_clustered import PriorBoxClusteredOp
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import PB


class TestPriorBoxClusteredExt(unittest.TestCase):
    @staticmethod
    def _create_priorbox_clustered_node(width=np.array([]), height=np.array([]),
                              flip=False, clip=False, variance=None, img_size=0, img_h=0,
                              img_w=0, step=0, step_h=0, step_w=0, offset=0):
        pb = onnx.helper.make_node(
            'PriorBoxClustered',
            inputs=['x'],
            outputs=['y'],
            width=width,
            height=height,
            flip=flip,
            clip=clip,
            variance=variance,
            img_size=img_size,
            img_h=img_h,
            img_w=img_w,
            step=step,
            step_h=step_h,
            step_w=step_w,
            offset=offset,
        )

        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['PriorBoxClustered'] = PriorBoxClusteredOp

    def test_priorbox_clustered_no_pb_no_ml(self):
        self.assertRaises(AttributeError, PriorBoxClusteredFrontExtractor.extract, None)

    def test_priorbox_clustered_ext_ideal_numbers(self):
        node = self._create_priorbox_clustered_node(width= np.array([2, 3], dtype=float),
                                          height=np.array([4, 5], dtype=float),
                                          variance=np.array([0.2, 0.3, 0.2, 0.3]),
                                          img_size=300, step=5.0, offset=0.6, flip=True)

        PriorBoxClusteredFrontExtractor.extract(node)

        exp_res = {
            'op': 'PriorBoxClustered',
            'type': 'PriorBoxClustered',
            'clip': 0,
            'flip': 1,
            'width': np.array([2, 3], dtype=float),
            'height': np.array([4, 5], dtype=float),
            'variance': [0.2, 0.3, 0.2, 0.3],
            'img_size': 300,
            'img_h': 0,
            'img_w': 0,
            'step': 5,
            'step_h': 0,
            'step_w': 0,
            'offset': 0.6
        }

        for key in exp_res.keys():
            if key in ['variance', 'width', 'height', 'step_h', 'step_w', 'offset']:
                np.testing.assert_almost_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])
