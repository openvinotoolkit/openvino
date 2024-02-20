# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.elementwise_ext import BiasToAdd
from unit_tests.utils.extractors import FakeModelLayer, FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakeBiasProtoLayer:
    def __init__(self, val):
        self.bias_param = val


class TestBias(unittest.TestCase):

    @patch('openvino.tools.mo.front.caffe.elementwise_ext.embed_input')
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
