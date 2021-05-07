# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from extensions.front.caffe.elu import ELUFrontExtractor
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode


class FakeProtoLayer:
    def __init__(self, val):
        self.elu_param = val


class TestElu(unittest.TestCase):
    @patch('extensions.front.caffe.elu.collect_attributes')
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
