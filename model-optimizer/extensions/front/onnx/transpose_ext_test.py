"""
 Copyright (C) 2018-2020 Intel Corporation

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

import itertools
import unittest

import numpy as np
import onnx
from generator import generator, generate

from extensions.front.onnx.transpose_ext import TransposeFrontExtractor
from extensions.ops.transpose import Transpose
from mo.ops.op import Op
from mo.utils.unittest.extractors import PB


@generator
class TestTransposeONNXExt(unittest.TestCase):
    @staticmethod
    def _create_transpose_node(order: list):
        if order is None:
            # Default transpose
            pb = onnx.helper.make_node(
                'Transpose',
                inputs=['data'],
                outputs=['transposed'],
            )
        else:
            # Transpose with order
            pb = onnx.helper.make_node(
                'Transpose',
                inputs=['data'],
                outputs=['transposed'],
                perm=order
            )
        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Transpose'] = Transpose
        pass

    # This generator generates all permutations for [0,1,2,3] and [0,1,2] orders
    @generate(*[list(order) for order in list(itertools.permutations(np.arange(4)))] +
               [list(order) for order in list(itertools.permutations(np.arange(3)))] + [None])
    def test_transpose_ext(self, order):
        node = self._create_transpose_node(order)
        TransposeFrontExtractor.extract(node)

        exp_res = {
            'type': 'Transpose',
            'order': order,
            'infer': Transpose.infer
        }

        for key in exp_res.keys():
            if isinstance(exp_res[key], list):
                self.assertTrue(np.array_equal(node[key], exp_res[key]),
                                "Orders are not the same: {} and {}".format(node[key], exp_res[key]))
            else:
                self.assertEqual(node[key], exp_res[key])
