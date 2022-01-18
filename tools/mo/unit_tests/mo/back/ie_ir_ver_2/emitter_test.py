# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock
from xml.etree.ElementTree import Element, tostring

import numpy as np

from openvino.tools.mo.back.ie_ir_ver_2.emitter import soft_get, xml_shape, serialize_runtime_info
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.runtime_info import RTInfo, OldAPIMapOrder, OldAPIMapElementType
from unit_tests.utils.graph import build_graph, result, regular_op

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


class TestSerializeRTInfo(unittest.TestCase):
    def test_serialize_old_api_map_parameter(self):
        graph = build_graph({**regular_op('placeholder', {'type': 'Parameter', 'rt_info': RTInfo()}),
                             **result('result')},
                            [('placeholder', 'result')], {}, nodes_with_edges_only=True)
        param_node = Node(graph, 'placeholder')
        param_node.rt_info.info[('old_api_map_order', 0)] = OldAPIMapOrder()
        param_node.rt_info.info[('old_api_map_order', 0)].old_api_transpose_parameter([0, 2, 3, 1])
        param_node.rt_info.info[('old_api_map_element_type', 0)] = OldAPIMapElementType()
        param_node.rt_info.info[('old_api_map_element_type', 0)].set_legacy_type(np.float32)

        net = Element('net')
        serialize_runtime_info(param_node, net)
        serialize_res = str(tostring(net))
        self.assertTrue("name=\"old_api_map_order\"" in serialize_res)
        self.assertTrue("name=\"old_api_map_element_type\"" in serialize_res)
        self.assertTrue("version=\"0\"" in serialize_res)
        self.assertTrue("value=\"0,2,3,1\"" in serialize_res)
        self.assertTrue("value=\"f32\"" in serialize_res)
        self.assertTrue(serialize_res.startswith("b'<net><rt_info>"))
        self.assertTrue(serialize_res.endswith("</rt_info></net>'"))

        del param_node.rt_info.info[('old_api_map_order', 0)]
        param_node.rt_info.info[('old_api_map_element_type', 0)] = OldAPIMapElementType()
        param_node.rt_info.info[('old_api_map_element_type', 0)].set_legacy_type(np.float16)

        net = Element('net')
        serialize_runtime_info(param_node, net)
        serialize_res = str(tostring(net))
        self.assertTrue("name=\"old_api_map_element_type\"" in serialize_res)
        self.assertTrue("version=\"0\"" in serialize_res)
        self.assertTrue("value=\"f16\"" in serialize_res)
        self.assertTrue(serialize_res.startswith("b'<net><rt_info>"))
        self.assertTrue(serialize_res.endswith("</rt_info></net>'"))

    def test_serialize_old_api_map_result(self):
        graph = build_graph({**regular_op('placeholder', {'type': 'Parameter', 'rt_info': RTInfo()}),
                             **regular_op('result', {'type': 'Result', 'rt_info': RTInfo()})},
                            [('placeholder', 'result')], {}, nodes_with_edges_only=True)
        result_node = Node(graph, 'result')
        result_node.rt_info.info[('old_api_map_order', 0)] = OldAPIMapOrder()
        result_node.rt_info.info[('old_api_map_order', 0)].old_api_transpose_result([0, 3, 1, 2])

        net = Element('net')
        serialize_runtime_info(result_node, net)
        serialize_res = str(tostring(net))
        self.assertTrue("name=\"old_api_map_order\"" in serialize_res)
        self.assertTrue("version=\"0\"" in serialize_res)
        self.assertTrue("value=\"0,3,1,2\"" in serialize_res)
        self.assertTrue(serialize_res.startswith("b'<net><rt_info>"))
        self.assertTrue(serialize_res.endswith("</rt_info></net>'"))
