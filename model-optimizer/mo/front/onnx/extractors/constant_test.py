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

import logging as log
import unittest

import numpy as np
import onnx
from generator import generator, generate
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from mo.front.onnx.extractors.constant import onnx_constant_ext
from mo.utils.unittest.extractors import PB

dtypes = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.float32, np.double, np.bool]


@generator
class ConstantONNXExtractorTest(unittest.TestCase):
    @staticmethod
    def _create_constant_node(numpy_dtype):
        numpy_dtype = np.dtype(numpy_dtype)
        if numpy_dtype not in NP_TYPE_TO_TENSOR_TYPE:
            log.error("Numpy type {} not supported in ONNX".format(numpy_dtype))
            return None

        values = np.array(np.random.randn(5, 5).astype(numpy_dtype))
        pb = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['values'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=NP_TYPE_TO_TENSOR_TYPE[numpy_dtype],
                dims=values.shape,
                vals=values.flatten().astype(numpy_dtype),
            ),
        )
        node = PB({'pb': pb})
        return node

    @generate(*dtypes)
    def test_constant_ext(self, np_dtype):
        node = self._create_constant_node(np_dtype)
        attrs = onnx_constant_ext(node)
        self.assertTrue(attrs['data_type'] == np_dtype,
                        'Wrong data_type attribute: recieved {}, expected {}'.format(attrs['data_type'], np_dtype))
