"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import onnx
import unittest

import numpy as np

from extensions.front.onnx.priorbox_clustered_ext import PriorBoxClusteredFrontExtractor
from extensions.ops.priorbox_clustered import PriorBoxClusteredOp
from mo.ops.op import Op
from mo.utils.unittest.extractors import PB


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
        node = self._create_priorbox_clustered_node(width= np.array([2, 3], dtype=np.float),
                                          height=np.array([4, 5], dtype=np.float),
                                          variance=np.array([0.2, 0.3, 0.2, 0.3]),
                                          img_size=300, step=5.0, offset=0.6, flip=True)

        PriorBoxClusteredFrontExtractor.extract(node)

        exp_res = {
            'op': 'PriorBoxClustered',
            'type': 'PriorBoxClustered',
            'clip': 0,
            'flip': 1,
            'width': np.array([2, 3], dtype=np.float),
            'height': np.array([4, 5], dtype=np.float),
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
