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

import onnx
from generator import generator, generate

from extensions.front.onnx.gather_ext import GatherFrontExtractor
from extensions.ops.gather import Gather
from mo.ops.op import Op
from mo.utils.unittest.extractors import PB


@generator
class TestGatherONNXExt(unittest.TestCase):
    @staticmethod
    def _create_gather_node(axis=0):
        pb = onnx.helper.make_node(
            'Gather',
            inputs=['data', 'indices'],
            outputs=['y'],
            axis=axis,
        )
        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Gather'] = Gather

    @generate(*[0, 1, 2, 3])
    def test_gather_ext(self, axis):
        node = self._create_gather_node(axis)
        GatherFrontExtractor.extract(node)

        exp_res = {
            'type': 'Gather',
            'axis': axis,
            'infer': Gather.infer
        }

        for key in exp_res.keys():
            self.assertEqual(node[key], exp_res[key])
