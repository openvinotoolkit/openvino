# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import tempfile
import numpy as np
from pathlib import Path

import openvino.runtime.opset11 as opset11
import openvino.runtime.opset10 as opset10
from openvino.runtime import Model, serialize

from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph
from openvino.tools.mo.utils.logger import init_logger

# required to be in global area to run MO IR Reader
init_logger('ERROR', False)


class TestOps(unittest.TestCase):
    @staticmethod
    def check_graph_can_save(model, name):
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / (name + '.xml')
            model_bin = Path(tmp) / (name + '.bin')
            serialize(model, model_xml, model_bin)
            graph, _ = restore_graph_from_ir(model_xml, model_bin)
            save_restored_graph(graph, tmp, {}, name)
            # restore 2 times to validate that after save graph doesn't lose attributes etc.
            graph, _ = restore_graph_from_ir(model_xml, model_bin)
            return graph

    def test_topk_11(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        k_val = np.int32(3)
        axis = np.int32(1)
        topk = opset11.topk(data_parameter, k_val, axis,
                            "max", "value", stable=True, name="TopK_11")
        model = Model(topk, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'topk_model')
        topk_node = graph.get_op_nodes(op="TopK")[0]
        self.assertEqual(topk_node["version"], "opset11")
        self.assertTrue(topk_node["stable"])

    def test_interpolate_11(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset11.interpolate(data_parameter, np.int32(
            [20, 48]), "nearest", "sizes", axes=np.int32([2, 3]), name="Interpolate_11")
        model = Model(interpolate, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'interpolate_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset11")

    def test_interpolate_4(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset10.interpolate(data_parameter, np.int32([20, 48]), np.float32(
            [2, 2]), "nearest", "sizes", axes=np.int32([2, 3]), name="Interpolate_4")
        model = Model(interpolate, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'interpolate4_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset4")

    def test_unique(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        unique = opset10.unique(data_parameter, axis=np.int32(
            [2]), sorted=True, name="Unique_10")
        model = Model(unique, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'unique_model')
        unique_node = graph.get_op_nodes(op="Unique")[0]
        self.assertEqual(unique_node["version"], "opset10")
        self.assertListEqual(unique_node.out_port(
            0).data.get_shape().tolist(), [6, 12, None, 24])
        self.assertTrue(unique_node["sorted"])

    def test_is_finite(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_finite = opset10.is_finite(data_parameter, name="Is_finite_10")
        model = Model(is_finite, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'is_finite_model')
        is_finite_node = graph.get_op_nodes(op="IsFinite")[0]
        self.assertEqual(is_finite_node["version"], "opset10")

    def test_is_inf(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_inf = opset10.is_inf(data_parameter, name="Is_inf_10")
        model = Model(is_inf, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'is_inf_model')
        is_inf_node = graph.get_op_nodes(op="IsInf")[0]
        self.assertEqual(is_inf_node["version"], "opset10")

    def test_is_nan(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_nan = opset10.is_nan(data_parameter, name="Is_nan_10")
        model = Model(is_nan, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'is_nan_model')
        is_nan_node = graph.get_op_nodes(op="IsNaN")[0]
        self.assertEqual(is_nan_node["version"], "opset10")
