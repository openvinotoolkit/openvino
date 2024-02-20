# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock

import defusedxml.ElementTree as ET
import numpy as np
from defusedxml import defuse_stdlib

from openvino.tools.mo.back.ie_ir_ver_2.emitter import soft_get, xml_shape, serialize_runtime_info, serialize_network, \
    port_renumber
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.infer import partial_infer, type_infer
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.ops.pooling import Pooling
from openvino.tools.mo.ops.result import Result
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.runtime_info import RTInfo, OldAPIMapOrder, OldAPIMapElementType
from openvino.tools.mo.utils.unsupported_ops import UnsupportedOps
from unit_tests.utils.graph import valued_const_with_data, result, regular_op_with_empty_data, connect, \
    shaped_parameter, build_graph, regular_op

# defuse_stdlib provide patched version of xml.etree.ElementTree which allows to use objects from xml.etree.ElementTree
# in a safe manner without including unsafe xml.etree.ElementTree
ET_defused = defuse_stdlib()[ET]
Element = ET_defused.Element
tostring = ET_defused.tostring

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


class TestSerialize(unittest.TestCase):
    @staticmethod
    def build_graph_with_gather():
        nodes = {
            **shaped_parameter('data', int64_array([3, 3]), {'data_type': np.float32, 'type': Parameter.op}),
            **shaped_parameter('indices', int64_array([1, 2]), {'data_type': np.float32, 'type': Parameter.op}),
            **valued_const_with_data('axis', int64_array(1)),
            **regular_op_with_empty_data('gather', {'op': 'Gather', 'batch_dims': 0, 'infer': Gather.infer,
                                                    'type': Gather.op}),
            **result('res'),
        }

        edges = [
            *connect('data', '0:gather'),
            *connect('indices', '1:gather'),
            *connect('axis', '2:gather'),
            *connect('gather', 'res'),
        ]

        graph = build_graph(nodes, edges)

        data_node = Node(graph, 'data')
        Parameter.update_node_stat(data_node, {})
        indices_node = Node(graph, 'indices')
        Parameter.update_node_stat(indices_node, {})

        gather_node = Node(graph, 'gather')
        Gather.update_node_stat(gather_node, {})

        res_node = Node(graph, 'res')
        Result.update_node_stat(res_node, {})

        partial_infer(graph)
        type_infer(graph)

        return graph

    @staticmethod
    def build_graph_with_maxpool():
        graph = build_graph(
            nodes_attrs={
                'input': {'kind': 'op', 'op': 'Parameter', 'name': 'node', 'infer': Parameter.infer,
                          'shape': [1, 3, 10, 10]},
                'input_data': {'kind': 'data', 'value': None, 'shape': None},

                'pool': {'kind': 'op', 'type': 'MaxPool', 'infer': Pooling.infer,
                         'window': np.array([1, 1, 2, 2]), 'stride': np.array([1, 1, 2, 2]),
                         'pad': np.array([[0, 0], [0, 0], [0, 0], [1, 1]]),
                         'pad_spatial_shape': np.array([[0, 0], [1, 1]]),
                         'pool_method': 'max', 'exclude_pad': False, 'global_pool': False,
                         'output_spatial_shape': None, 'output_shape': None,
                         'kernel_spatial': np.array([2, 2]), 'spatial_dims': np.array([2, 3]),
                         'channel_dims': np.array([1]), 'batch_dims': np.array([0]),
                         'pooling_convention': 'full', 'dilation': np.array([1, 1, 2, 2]),
                         'auto_pad': 'valid'},

                'pool_data': {'kind': 'data', 'value': None, 'shape': None},
                'pool_data_added': {'kind': 'data', 'value': None, 'shape': None},
                'result': {'kind': 'op', 'op': 'Result'},
                'result_added': {'kind': 'op', 'op': 'Result'}
            },
            edges=[
                ('input', 'input_data'),
                ('input_data', 'pool'),
                ('pool', 'pool_data', {'out': 0}),
                ('pool_data', 'result'),
                ('pool', 'pool_data_added', {'out': 1}),
                ('pool_data_added', 'result_added')
            ]
        )

        input_node = Node(graph, 'input')
        Parameter.update_node_stat(input_node, {})

        pool_node = Node(graph, 'pool')
        Pooling.update_node_stat(pool_node, {'pool_method': 'max'})

        result_node = Node(graph, 'result')
        Result.update_node_stat(result_node, {})
        result_added_node = Node(graph, 'result_added')
        Result.update_node_stat(result_added_node, {})

        partial_infer(graph)
        type_infer(graph)
        return graph

    def test_gather(self):
        graph = self.build_graph_with_gather()

        net = Element('net')
        graph.outputs_order = ['gather']
        unsupported = UnsupportedOps(graph)
        port_renumber(graph)

        serialize_network(graph, net, unsupported)
        xml_string = str(tostring(net))
        self.assertTrue("type=\"Parameter\"" in xml_string)
        self.assertTrue("type=\"Result\"" in xml_string)
        self.assertTrue("type=\"Gather\"" in xml_string)

    def test_maxpool(self):
        graph = self.build_graph_with_maxpool()

        net = Element('net')
        graph.outputs_order = ['pool']
        unsupported = UnsupportedOps(graph)
        port_renumber(graph)
        serialize_network(graph, net, unsupported)
        xml_string = str(tostring(net))
        self.assertTrue("type=\"Parameter\"" in xml_string)
        self.assertTrue("type=\"Result\"" in xml_string)
        self.assertTrue("type=\"Pooling\"" in xml_string)

    def test_maxpool_raises(self):
        graph = self.build_graph_with_maxpool()

        pool_node = Node(graph, 'pool')
        result_node = Node(graph, 'result')
        result_added_node = Node(graph, 'result_added')
        pool_out_1 = Node(graph, 'pool_data')
        pool_out_2 = Node(graph, 'pool_data_added')

        # when operation does not have output data nodes Exception should be raised
        graph.remove_edge(pool_node.id, pool_out_1.id)
        graph.remove_edge(pool_node.id, pool_out_2.id)
        graph.remove_edge(pool_out_1.id, result_node.id)
        graph.remove_edge(pool_out_2.id, result_added_node.id)

        graph.remove_node(result_node.id)
        graph.remove_node(result_added_node.id)

        net = Element('net')
        graph.outputs_order = ['pool']
        unsupported = UnsupportedOps(graph)
        port_renumber(graph)

        with self.assertRaisesRegex(AssertionError, "Incorrect graph. Non-Result node.*"):
            serialize_network(graph, net, unsupported)
