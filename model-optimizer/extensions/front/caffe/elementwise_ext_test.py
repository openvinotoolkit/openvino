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
import unittest
from unittest.mock import patch

from extensions.front.caffe.elementwise_ext import BiasToAdd
from mo.utils.unittest.extractors import FakeModelLayer, FakeMultiParam
from mo.utils.unittest.graph import FakeNode


class FakeBiasProtoLayer:
    def __init__(self, val):
        self.bias_param = val


class TestBias(unittest.TestCase):

    @patch('extensions.front.caffe.elementwise_ext.embed_input')
    def test_bias(self, embed_input_mock):
        embed_input_mock.return_value = {}
        params = {'axis': 1}
        add_node = FakeNode(FakeBiasProtoLayer(FakeMultiParam(params)),
                            FakeModelLayer([1, 2, 3, 4, 5]))
        BiasToAdd.extract(add_node)

        exp_res = {
            'type': "Add",
            'axis': 1
        }

        for key in exp_res.keys():
            self.assertEqual(add_node[key], exp_res[key])
