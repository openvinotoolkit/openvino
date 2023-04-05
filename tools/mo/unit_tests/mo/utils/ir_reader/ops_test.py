# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import tempfile
import numpy as np
from pathlib import Path
from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph

import openvino.runtime.opset11 as opset11
import openvino.runtime.opset10 as opset10
from openvino.runtime import Model, serialize


class TestOps(unittest.TestCase):
    def test_topk_11(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        k_val = np.int32(3)
        axis = np.int32(1)
        topk = opset11.topk(data_parameter, k_val, axis,
                            "max", "value", stable=True, name="TopK_11")
        model = Model(topk, [data_parameter])
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / 'topk_model.xml'
            model_bin = Path(tmp) / 'topk_model.bin'
            serialize(model, model_xml, model_bin)
            graph, _ = restore_graph_from_ir(model_xml, model_bin)
            topk_node = graph.nodes()["TopK_11"]
            self.assertEqual(topk_node["version"], "opset11")
            self.assertTrue(topk_node["stable"])

    def test_interpolate_11(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset11.interpolate(data_parameter, np.int32(
            [20, 48]), "nearest", "sizes", axes=np.int32([2, 3]), name="Interpolate_11")
        model = Model(interpolate, [data_parameter])
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / 'interpolate_model.xml'
            model_bin = Path(tmp) / 'interpolate_model.bin'
            serialize(model, model_xml, model_bin)
            graph, _ = restore_graph_from_ir(model_xml, model_bin)
            interpolate_node = graph.nodes()["Interpolate_11"]
            self.assertEqual(interpolate_node["version"], "opset11")

    def test_interpolate_4(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset10.interpolate(data_parameter, np.int32([20, 48]), np.float32(
            [2, 2]), "nearest", "sizes", axes=np.int32([2, 3]), name="Interpolate_4")
        model = Model(interpolate, [data_parameter])
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / 'interpolate_model.xml'
            model_bin = Path(tmp) / 'interpolate_model.bin'
            serialize(model, model_xml, model_bin)
            graph, _ = restore_graph_from_ir(model_xml, model_bin)
            interpolate_node = graph.nodes()["Interpolate_4"]
            self.assertEqual(interpolate_node["version"], "opset4")

    def test_unique(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        unique = opset10.unique(data_parameter, sorted=True, name="Unique_10")
        res = opset10.result(unique.output(0), name="output")
        model = Model(res, [data_parameter])
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / 'unique_model.xml'
            model_bin = Path(tmp) / 'unique_model.bin'
            serialize(model, model_xml, model_bin)
            graph, _ = restore_graph_from_ir(model_xml, model_bin)
            unique_node = graph.nodes()["Unique_10"]
            self.assertEqual(unique_node["version"], "opset10")
            self.assertTrue(unique_node["sorted"])

    def test_is_finite(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_finite = opset10.is_finite(data_parameter, name="Is_finite_10")
        model = Model(is_finite, [data_parameter])
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / 'is_finite_model.xml'
            model_bin = Path(tmp) / 'is_finite_model.bin'
            serialize(model, model_xml, model_bin)
            graph, meta = restore_graph_from_ir(model_xml, model_bin)
            is_finite_node = graph.nodes()["Is_finite_10"]
            self.assertEqual(is_finite_node["version"], "opset10")
            save_restored_graph(graph, tmp, meta, "is_finite_model_after")

    def test_is_inf(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_inf = opset10.is_inf(data_parameter, name="Is_inf_10")
        model = Model(is_inf, [data_parameter])
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / 'is_inf_model.xml'
            model_bin = Path(tmp) / 'is_inf_model.bin'
            serialize(model, model_xml, model_bin)
            graph, meta = restore_graph_from_ir(model_xml, model_bin)
            is_inf_node = graph.nodes()["Is_inf_10"]
            self.assertEqual(is_inf_node["version"], "opset10")
            save_restored_graph(graph, tmp, meta, "is_inf_model_after")

    def test_is_nan(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_nan = opset10.is_nan(data_parameter, name="Is_nan_10")
        model = Model(is_nan, [data_parameter])
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / 'is_nan_model.xml'
            model_bin = Path(tmp) / 'is_nan_model.bin'
            serialize(model, model_xml, model_bin)
            graph, meta = restore_graph_from_ir(model_xml, model_bin)
            is_nan_node = graph.nodes()["Is_nan_10"]
            self.assertEqual(is_nan_node["version"], "opset10")
            save_restored_graph(graph, tmp, meta, "is_nan_model_after")
