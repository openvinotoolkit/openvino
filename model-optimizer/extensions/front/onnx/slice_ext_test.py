"""
 Copyright (c) 2018 Intel Corporation

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

from extensions.front.onnx.slice_ext import SliceFrontExtractor
from mo.ops.op import Op
from mo.ops.slice import Slice
from mo.utils.unittest.extractors import PB


@generator
class TestSliceONNXExt(unittest.TestCase):
    @staticmethod
    def _create_slice_node(axes, starts, ends):
        if axes is None:
            pb = onnx.helper.make_node(
                'Slice',
                inputs=['x'],
                outputs=['y'],
                starts=starts,
                ends=ends,
            )
        else:
            pb = onnx.helper.make_node(
                'Slice',
                inputs=['x'],
                outputs=['y'],
                axes=axes,
                starts=starts,
                ends=ends,
            )

        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Slice'] = Slice

    @generate(*[([0, 1], [0, 0], [28, 28]), (None, [0, 0], [28, 28])])
    def test_slice_ext(self, axes, starts, ends):
        node = self._create_slice_node(axes, starts, ends)
        SliceFrontExtractor.extract(node)

        exp_res = {
            'op': 'Slice',
            'axis': axes,
            'start': starts,
            'end': ends,
            'infer': Slice.infer
        }

        for key in exp_res.keys():
            if type(node[key]) in [list, np.ndarray]:
                self.assertTrue(np.array_equal(np.array(node[key]), np.array(exp_res[key])))
            else:
                self.assertEqual(node[key], exp_res[key])
