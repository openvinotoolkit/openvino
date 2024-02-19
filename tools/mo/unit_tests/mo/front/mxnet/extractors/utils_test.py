# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import mxnet as mx

from openvino.tools.mo.front.mxnet.extractors.utils import AttrDictionary
from openvino.tools.mo.front.mxnet.extractors.utils import load_params


class TestAttrDictionary(unittest.TestCase):
    def testBool(self):
        attrs = {
            "global_pool": "True"
        }

        attr_dict = AttrDictionary(attrs)
        global_pool = attr_dict.bool("global_pool", False)
        self.assertEqual(True, global_pool)

    def testBoolAsDigits(self):
        attrs = {
            "global_pool": "1"
        }

        attr_dict = AttrDictionary(attrs)
        global_pool = attr_dict.bool("global_pool", False)
        self.assertEqual(True, global_pool)

    def testBoolWithoutAttr(self):
        attrs = {
            "something": "1"
        }

        attr_dict = AttrDictionary(attrs)
        global_pool = attr_dict.bool("global_pool", False)
        self.assertEqual(False, global_pool)

    def testStrAttr(self):
        attrs = {
            "something": "Val"
        }

        attr_dict = AttrDictionary(attrs)
        attr = attr_dict.str("something", "Text")
        self.assertEqual("Val", attr)

    def testStrAttrWithoutAttr(self):
        attrs = {
            "something2": "Val"
        }

        attr_dict = AttrDictionary(attrs)
        attr = attr_dict.str("something", "Text")
        self.assertEqual("Text", attr)

    def testFloatAttr(self):
        attrs = {
            "something": "0.5"
        }

        attr_dict = AttrDictionary(attrs)
        attr = attr_dict.float("something", 0.1)
        self.assertEqual(0.5, attr)

    def testFloatWithoutAttr(self):
        attrs = {
            "something2": "0.5"
        }

        attr_dict = AttrDictionary(attrs)
        attr = attr_dict.float("something", 0.1)
        self.assertEqual(0.1, attr)

    def testIntAttr(self):
        attrs = {
            "something": "5"
        }

        attr_dict = AttrDictionary(attrs)
        attr = attr_dict.float("something", 1)
        self.assertEqual(5, attr)

    def testIntWithoutAttr(self):
        attrs = {
            "something2": "5"
        }

        attr_dict = AttrDictionary(attrs)
        attr = attr_dict.float("something", 1)
        self.assertEqual(1, attr)

    def testTupleAttr(self):
        attrs = {
            "something": "(5,6,7)"
        }

        attr_dict = AttrDictionary(attrs)
        a, b, c = attr_dict.tuple("something", int, (1, 2, 3))
        self.assertEqual(5, a)
        self.assertEqual(6, b)
        self.assertEqual(7, c)

    def testTupleWithoutAttr(self):
        attrs = {
            "something2": "(5,6,7)"
        }

        attr_dict = AttrDictionary(attrs)
        a, b, c = attr_dict.tuple("something", int, (1, 2, 3))
        self.assertEqual(1, a)
        self.assertEqual(2, b)
        self.assertEqual(3, c)

    def testTupleWithEmptyTupleAttr(self):
        attrs = {
            "something2": "()"
        }

        attr_dict = AttrDictionary(attrs)
        a, b = attr_dict.tuple("something", int, (2, 3))
        self.assertEqual(2, a)
        self.assertEqual(3, b)

    def testTupleWithEmptyListAttr(self):
        attrs = {
            "something2": "[]"
        }

        attr_dict = AttrDictionary(attrs)
        a, b = attr_dict.tuple("something", int, (2, 3))
        self.assertEqual(2, a)
        self.assertEqual(3, b)

    def testListAttr(self):
        attrs = {
            "something": "5,6,7"
        }

        attr_dict = AttrDictionary(attrs)
        l = attr_dict.list("something", int, [1, 2, 3])
        self.assertEqual(5, l[0])
        self.assertEqual(6, l[1])
        self.assertEqual(7, l[2])

    def testListWithoutAttr(self):
        attrs = {
            "something2": "5,6,7"
        }

        attr_dict = AttrDictionary(attrs)
        l = attr_dict.list("something", int, [1, 2, 3])
        self.assertEqual(1, l[0])
        self.assertEqual(2, l[1])
        self.assertEqual(3, l[2])

    def testIntWithAttrNone(self):
        attrs = {
            "something": "None"
        }

        attr_dict = AttrDictionary(attrs)
        attr = attr_dict.int("something", None)
        self.assertEqual(None, attr)


class TestUtils(unittest.TestCase):
    @patch('mxnet.nd.load')
    def test_load_symbol_nodes_from_params(self, mock_nd_load):
        mock_nd_load.return_value = {'arg:conv0_weight': mx.nd.array([1, 2], dtype='float32'),
                                     'arg:conv1_weight': mx.nd.array([2, 3], dtype='float32'),
                                     'aux:bn_data_mean': mx.nd.array([5, 6], dtype='float32')}
        model_params = load_params("model.params")
        self.assertTrue('conv0_weight' in model_params._param_names)
        self.assertTrue('conv1_weight' in model_params._param_names)
        self.assertTrue('bn_data_mean' in model_params._aux_names)
        self.assertEqual([1., 2.], model_params._arg_params['conv0_weight'].asnumpy().tolist())
        self.assertEqual([2., 3.], model_params._arg_params['conv1_weight'].asnumpy().tolist())
        self.assertEqual([5., 6.], model_params._aux_params['bn_data_mean'].asnumpy().tolist())

    @patch('mxnet.nd.load')
    def test_load_symbol_nodes_from_args_nd(self, mock_nd_load):
        mock_nd_load.return_value = {'conv0_weight': mx.nd.array([1, 2], dtype='float32'),
                                     'conv1_weight': mx.nd.array([2, 3], dtype='float32')}
        model_params = load_params("args_model.nd", data_names=('data1', 'data2'))
        self.assertTrue('conv0_weight' in model_params._param_names)
        self.assertTrue('conv1_weight' in model_params._param_names)
        self.assertEqual([1., 2.], model_params._arg_params['conv0_weight'].asnumpy().tolist())
        self.assertEqual([2., 3.], model_params._arg_params['conv1_weight'].asnumpy().tolist())

    @patch('mxnet.nd.load')
    def test_load_symbol_nodes_from_auxs_nd(self, mock_nd_load):
        mock_nd_load.return_value = {'bn_data_mean': mx.nd.array([5, 6], dtype='float32')}
        model_params = load_params("auxs_model.nd")
        self.assertTrue('bn_data_mean' in model_params._aux_names)
        self.assertEqual([5., 6.], model_params._aux_params['bn_data_mean'].asnumpy().tolist())
