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

from extensions.front.onnx.image_scaler_ext import ImageScalerFrontExtractor
from mo.utils.unittest.extractors import PB


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
