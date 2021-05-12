# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock
from xml.etree.ElementTree import Element, tostring

import numpy as np

from mo.back.ie_ir_ver_2.emitter import soft_get, xml_shape
from mo.utils.error import Error

expected_result = b'<net><dim>2</dim><dim>10</dim><dim>50</dim><dim>50</dim></net>'


class TestEmitter(unittest.TestCase):
    def test_xml_shape(self):
        net = Element('net')
        xml_shape(np.array([2, 10, 50, 50], dtype=np.int64), net)
        self.assertEqual(tostring(net), expected_result)

    def test_xml_shape_float_values(self):
        net = Element('net')
        xml_shape(np.array([2.0, 10.0, 50.0, 50.0], dtype=np.float32), net)
        self.assertEqual(tostring(net), expected_result)

    def test_xml_shape_non_integer_values(self):
        net = Element('net')
        with self.assertRaises(Error):
            xml_shape(np.array([2.0, 10.0, 50.0, 50.5], dtype=np.float32), net)

    def test_xml_shape_negative_values(self):
        net = Element('net')
        with self.assertRaises(Error):
            xml_shape(np.array([2, 10, 50, -50], dtype=np.int64), net)


class TestSoftGet(unittest.TestCase):

    def test_node(self):
        node = MagicMock()
        node.soft_get = lambda attr: attr
        self.assertEqual(soft_get(node, 'string'), 'string')

    def test_not_callable(self):
        node = MagicMock()
        node.soft_get = 'foo'
        self.assertEqual(soft_get(node, 'string'), '<SUB-ELEMENT>')

    def test_not_node_1(self):
        node = {'soft_get': lambda attr: attr}
        self.assertEqual(soft_get(node, 'string'), '<SUB-ELEMENT>')

    def test_not_node_2(self):
        node = 'something-else'
        self.assertEqual(soft_get(node, 'string'), '<SUB-ELEMENT>')
