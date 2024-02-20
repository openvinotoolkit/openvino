# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.elu import ELUFrontExtractor
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakeProtoLayer:
    def __init__(self, val):
        self.elu_param = val


class TestElu(unittest.TestCase):
    @patch('openvino.tools.mo.front.caffe.elu.collect_attributes')
    def test_elu_ext(self, collect_attrs_mock):
        params = {
            'alpha': 4
        }
        collect_attrs_mock.return_value = {
            **params,
            'test': 54,
            'test2': 'test3'
        }

        fn = FakeNode(FakeProtoLayer(FakeMultiParam(params)), None)
        ELUFrontExtractor.extract(fn)

        exp_res = {
            'type': 'Elu',
            'alpha': 4
        }

        for i in exp_res:
            self.assertEqual(fn[i], exp_res[i])
