"""
 Copyright (c) 2018-2019 Intel Corporation

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
