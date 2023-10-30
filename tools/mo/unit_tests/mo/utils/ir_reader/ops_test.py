# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import tempfile
import numpy as np
from pathlib import Path

import openvino.runtime.opset13 as opset13
import openvino.runtime.opset12 as opset12
import openvino.runtime.opset11 as opset11
import openvino.runtime.opset10 as opset10
from openvino.runtime import Model, serialize, Core, PartialShape, Dimension, Type

from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph
from openvino.tools.mo.utils.logger import init_logger

# required to be in global area to run MO IR Reader
init_logger('ERROR', False)


class TestOps(unittest.TestCase):
    @staticmethod
    def check_graph_can_save(model, name):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_xml = tmp_path / (name + '.xml')
            model_bin = tmp_path / (name + '.bin')
            serialize(model, model_xml, model_bin)
            graph, _ = restore_graph_from_ir(model_xml, model_bin)
            save_restored_graph(graph, tmp, {}, name + '_restored')
            # restore 2 times to validate that after save graph doesn't lose attributes etc.
            restored_model_xml = tmp_path / (name + '_restored.xml')
            restored_model_bin = tmp_path / (name + '_restored.bin')
            graph, _ = restore_graph_from_ir(
                restored_model_xml, restored_model_bin)
            core = Core()
            core.set_property({"ENABLE_MMAP": False})
            # check that re-saved model can be read in runtime
            model = core.read_model(restored_model_xml)
            return graph, model

    def test_topk_11(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        k_val = np.int32(3)
        axis = np.int32(1)
        topk = opset11.topk(data_parameter, k_val, axis,
                            "max", "value", stable=True, name="TopK_11")
        model = Model(topk, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'topk_model')
        topk_node = graph.get_op_nodes(op="TopK")[0]
        self.assertEqual(topk_node["version"], "opset11")
        self.assertTrue(topk_node["stable"])
        self.assertEqual(topk_node["index_element_type"], np.int32)

    def test_interpolate_11(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset11.interpolate(data_parameter, np.int32(
            [20, 48]), "nearest", "sizes", axes=np.int32([2, 3]), name="Interpolate_11")
        model = Model(interpolate, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'interpolate_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset11")
        self.assertTrue("force_precision_in_ports" in interpolate_node)
        self.assertEqual(interpolate_node["force_precision_in_ports"], {1: 'int64'})

    def test_interpolate_11_scales(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset11.interpolate(data_parameter, np.float32(
            [2., 2.]), "nearest", "scales", axes=np.int32([2, 3]), name="Interpolate_11")
        model = Model(interpolate, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'interpolate_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset11")
        self.assertTrue("force_precision_in_ports" not in interpolate_node)

    def test_interpolate_11_no_axes(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset11.interpolate(data_parameter, np.int32(
            [6, 12, 20, 48]), "nearest", "sizes", name="Interpolate_11")
        model = Model(interpolate, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'interpolate_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset11")
        self.assertTrue("force_precision_in_ports" in interpolate_node)
        self.assertEqual(interpolate_node["force_precision_in_ports"], {1: 'int64'})

    def test_interpolate_4(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset10.interpolate(data_parameter, np.int32([20, 48]), np.float32(
            [2, 2]), "nearest", "sizes", axes=np.int32([2, 3]), name="Interpolate_4")
        model = Model(interpolate, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'interpolate4_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset4")

    def test_unique(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        unique = opset10.unique(data_parameter, axis=np.int32(
            [2]), sorted=True, name="Unique_10")
        model = Model(unique, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'unique_model')
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
        graph, _ = TestOps.check_graph_can_save(model, 'is_finite_model')
        is_finite_node = graph.get_op_nodes(op="IsFinite")[0]
        self.assertEqual(is_finite_node["version"], "opset10")

    def test_is_inf(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_inf = opset10.is_inf(data_parameter, name="Is_inf_10")
        model = Model(is_inf, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'is_inf_model')
        is_inf_node = graph.get_op_nodes(op="IsInf")[0]
        self.assertEqual(is_inf_node["version"], "opset10")

    def test_is_nan(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_nan = opset10.is_nan(data_parameter, name="Is_nan_10")
        model = Model(is_nan, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'is_nan_model')
        is_nan_node = graph.get_op_nodes(op="IsNaN")[0]
        self.assertEqual(is_nan_node["version"], "opset10")

    def test_if(self):
        parameter_x = opset11.parameter([2], np.float32, "pX")
        parameter_y = opset11.parameter([2], np.float32, "pY")
        const_z = opset11.constant(4.0, dtype=np.float32)

        condition = opset11.constant(True, dtype=bool)

        # then_body
        x_t = opset11.parameter([2], np.float32, "X")
        y_t = opset11.parameter([2], np.float32, "Y")
        mmul_t = opset11.matmul(x_t, y_t, False, False)
        mul_t = opset11.multiply(y_t, x_t)
        then_body_res_1 = opset11.result(mmul_t)
        then_body_res_2 = opset11.result(mul_t)
        then_body = Model([then_body_res_1, then_body_res_2], [x_t, y_t])

        # else_body
        x_e = opset11.parameter([2], np.float32, "X")
        z_e = opset11.parameter([], np.float32, "Z")
        mul_e = opset11.multiply(x_e, z_e)
        else_body_res_1 = opset11.result(z_e)
        else_body_res_2 = opset11.result(mul_e)
        else_body = Model([else_body_res_1, else_body_res_2], [x_e, z_e])

        if_node = opset11.if_op(condition)
        if_node.set_friendly_name("If_opset8")
        if_node.set_then_body(then_body)
        if_node.set_else_body(else_body)
        if_node.set_input(parameter_x.output(0), x_t, x_e)
        if_node.set_input(parameter_y.output(0), y_t, None)
        if_node.set_input(const_z.output(0), None, z_e)
        out1 = if_node.set_output(then_body_res_1, else_body_res_1)
        out2 = if_node.set_output(then_body_res_2, else_body_res_2)

        model = Model([out1, out2], [parameter_x, parameter_y])
        graph, _ = TestOps.check_graph_can_save(model, 'if_model')
        if_node = graph.get_op_nodes(op="If")[0]
        self.assertEqual(if_node["version"], "opset8")
        _, layer_info, _ = if_node['IE'][0]
        _, callable_attribute = layer_info[0]
        self.assertTrue(callable(callable_attribute))
        self.assertEqual(callable_attribute(if_node), "If_opset8")

    def test_strided_slice_no_begin_end_mask(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        strided_slice = opset11.strided_slice(data_parameter, np.int32([1, 2, 3, 4]), np.int32(
            [3, 6, 9, 12]), np.int32([1, 1, 1, 1]), begin_mask=[], end_mask=[], name="StridedSlice_10")
        model = Model(strided_slice, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'strided_slice_model')
        strided_slice_node = graph.get_op_nodes(op="StridedSlice")[0]
        self.assertEqual(strided_slice_node["version"], "opset1")

    def test_scatter_dynamic_shape(self):
        data_parameter = opset11.parameter(
            PartialShape.dynamic(Dimension(2)), name="Data", dtype=np.float32)
        shape_of = opset11.shape_of(data_parameter)
        gather = opset11.gather(shape_of, np.int32(1), 0)
        unsqueeze = opset11.unsqueeze(gather, 0)
        scatter = opset11.scatter_update(np.int64([0, 0]), np.int64([1]), unsqueeze, axis=0)
        mul = opset11.multiply(scatter, np.int64([1, 2]))
        reshape = opset11.reshape(data_parameter, mul, True)
        model = Model(reshape, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'scatter_dynamic_model')
        scatter_update_node = graph.get_op_nodes(op="ScatterUpdate")[0]
        self.assertListEqual(scatter_update_node.out_port(0).data.get_value().tolist(), [0, None])

    def test_pad_12(self):
        data_parameter = opset12.parameter([6, 12, 10, 24], name="Data", dtype=np.float32)
        pad = opset12.pad(data_parameter, np.int64([0, 0, -1, -2]), np.int64([0, 0, -3, -4]), "constant")
        model = Model(pad, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'pad_model')
        pad_node = graph.get_op_nodes(op="Pad")[0]
        self.assertEqual(pad_node["version"], "opset12")
        self.assertListEqual(pad_node.in_port(1).data.get_value().tolist(), [0, 0, -1, -2])
        self.assertListEqual(pad_node.in_port(2).data.get_value().tolist(), [0, 0, -3, -4])
        self.assertListEqual(pad_node.out_port(0).data.get_shape().tolist(), [6, 12, 6, 18])

    def test_scatter_elements_update_12(self):
        data_parameter = opset12.parameter([10], name="Data", dtype=np.float32)
        scatter = opset12.scatter_elements_update(data_parameter, np.int32([5, 0, 7, 5]), np.float32([5., 6., 1.5, -5.]), np.int32(0), "sum", False)
        model = Model(scatter, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'scatter_model')
        scatter_node = graph.get_op_nodes(op="ScatterElementsUpdate")[0]
        self.assertListEqual(scatter_node.out_port(0).data.get_shape().tolist(), [10])
        self.assertEqual(scatter_node["version"], "opset12")
        self.assertEqual(scatter_node['reduction'], 'sum')
        self.assertFalse(scatter_node['use_init_val'])

    def test_group_norm_12(self):
        data_parameter = opset12.parameter([1, 3, 3, 3], name="Data", dtype=np.float32)
        scale = np.array((1, 1, 1), dtype=np.float32)
        bias = np.array((1, 1, 1), dtype=np.float32)
        num_groups = 1
        epsilon = 1e-6
        node = opset12.group_normalization(data_parameter, scale, bias, num_groups, epsilon)
        model = Model(node, [data_parameter])
        graph, _ = TestOps.check_graph_can_save(model, 'group_norm_model')
        gn_node = graph.get_op_nodes(op="GroupNormalization")[0]
        self.assertListEqual(gn_node.out_port(0).data.get_shape().tolist(), [1, 3, 3, 3])
        self.assertEqual(gn_node["version"], "opset12")
        self.assertEqual(gn_node['num_groups'], 1)
        self.assertEqual(gn_node['epsilon'], 1e-06)

    def test_bitwise_and_13(self):
        a = opset13.parameter([4, 1], name="A", dtype=np.int32)
        b = opset13.parameter([1, 2], name="B", dtype=np.int32)

        op = opset13.bitwise_and(a, b)
        model = Model(op, [a, b])
        graph, _ = TestOps.check_graph_can_save(model, "bitwise_and_model")
        op_node = graph.get_op_nodes(op="BitwiseAnd")[0]
        self.assertListEqual(op_node.out_port(0).data.get_shape().tolist(), [4, 2])
        self.assertEqual(op_node["version"], "opset13")
        self.assertEqual(op_node["auto_broadcast"], "numpy")

    def test_bitwise_or_13(self):
        a = opset13.parameter([4, 1], name="A", dtype=np.int32)
        b = opset13.parameter([1, 2], name="B", dtype=np.int32)

        op = opset13.bitwise_or(a, b)
        model = Model(op, [a, b])
        graph, _ = TestOps.check_graph_can_save(model, "bitwise_or_model")
        op_node = graph.get_op_nodes(op="BitwiseOr")[0]
        self.assertListEqual(op_node.out_port(0).data.get_shape().tolist(), [4, 2])
        self.assertEqual(op_node["version"], "opset13")
        self.assertEqual(op_node["auto_broadcast"], "numpy")

    def test_bitwise_xor_13(self):
        a = opset13.parameter([4, 1], name="A", dtype=np.int32)
        b = opset13.parameter([1, 2], name="B", dtype=np.int32)

        op = opset13.bitwise_xor(a, b)
        model = Model(op, [a, b])
        graph, _ = TestOps.check_graph_can_save(model, "bitwise_xor_model")
        op_node = graph.get_op_nodes(op="BitwiseXor")[0]
        self.assertListEqual(op_node.out_port(0).data.get_shape().tolist(), [4, 2])
        self.assertEqual(op_node["version"], "opset13")
        self.assertEqual(op_node["auto_broadcast"], "numpy")

    def test_bitwise_not_13(self):
        a = opset13.parameter([4, 2], name="A", dtype=np.int32)

        op = opset13.bitwise_not(a)
        model = Model(op, [a])
        graph, _ = TestOps.check_graph_can_save(model, "bitwise_not_model")
        op_node = graph.get_op_nodes(op="BitwiseNot")[0]
        self.assertListEqual(op_node.out_port(0).data.get_shape().tolist(), [4, 2])
        self.assertEqual(op_node["version"], "opset13")

    def test_multinomial_13_param_inputs(self):
        data_shape = [2, 8]
        probs = opset13.parameter(
            data_shape, name="probs", dtype=np.float32)
        num_samples = opset13.parameter(
            [1], name="num_samples", dtype=np.int32)

        op = opset13.multinomial(probs, num_samples,
                                 convert_type="i32",
                                 with_replacement=True,
                                 log_probs=True,
                                 global_seed=456,
                                 op_seed=213)

        model = Model(op, [probs, num_samples])
        graph, loaded_model = TestOps.check_graph_can_save(
            model, 'multinomial_param_model')
        graph_node = graph.get_op_nodes(op="Multinomial")[0]

        self.assertEqual(graph_node["version"], "opset13")
        self.assertListEqual(graph_node.out_port(
            0).data.get_shape().tolist(), [2, None])
        self.assertEqual(graph_node["convert_type"], "i32")
        self.assertTrue(graph_node["with_replacement"])
        self.assertTrue(graph_node["log_probs"])
        self.assertEqual(graph_node["global_seed"], 456)
        self.assertEqual(graph_node["op_seed"], 213)
        self.assertEqual(loaded_model.get_output_element_type(0), Type.i32)
        self.assertEqual(loaded_model.get_output_partial_shape(
            0), PartialShape([2, -1]))

    def test_multinomial_13_const_inputs(self):
        probs = opset13.constant(
            [[0.4, 0.5, 0.1], [0.3, 0.2, 0.5]], name="probs", dtype=np.float32)
        num_samples = opset13.constant(
            [3], name="num_samples", dtype=np.int64)

        op = opset13.multinomial(probs, num_samples,
                                 convert_type="i64",
                                 with_replacement=False,
                                 log_probs=False)

        model = Model(op, [])
        graph, loaded_model = TestOps.check_graph_can_save(
            model, 'multinomial_const_model')
        graph_node = graph.get_op_nodes(op="Multinomial")[0]

        self.assertEqual(graph_node["version"], "opset13")
        self.assertListEqual(graph_node.out_port(
            0).data.get_shape().tolist(), [2, 3])
        self.assertEqual(graph_node["convert_type"], "i64")
        self.assertFalse(graph_node["with_replacement"])
        self.assertFalse(graph_node["log_probs"])
        self.assertEqual(graph_node["global_seed"], 0)
        self.assertEqual(graph_node["op_seed"], 0)
        self.assertEqual(loaded_model.get_output_element_type(0), Type.i64)
        self.assertEqual(loaded_model.get_output_partial_shape(
            0), PartialShape([2, 3]))

    def test_nms_rotated_13_attrs_false_i32(self):
        boxes_shape = [1, 100, 5]
        scores_shape = [1, 2, 100]
        max_output_boxes_val = 5
        iou_threshold_val = 0.5
        score_threshold_val = 0.4

        boxes_parameter = opset13.parameter(
            boxes_shape, name="Boxes", dtype=np.float32)
        scores_parameter = opset13.parameter(
            scores_shape, name="Scores", dtype=np.float32)

        max_output_boxes = opset13.constant([max_output_boxes_val], np.int64)
        iou_threshold = opset13.constant([iou_threshold_val], np.float32)
        score_threshold = opset13.constant([score_threshold_val], np.float32)

        sort_result_descending = False
        output_type = "i32"
        clockwise = False

        node = opset13.nms_rotated(boxes_parameter, scores_parameter, max_output_boxes, iou_threshold,
                                   score_threshold, sort_result_descending, output_type, clockwise)

        model = Model(node, [boxes_parameter, scores_parameter])
        graph, loaded_model = TestOps.check_graph_can_save(
            model, 'nms_rotated_model_1')
        ir_node = graph.get_op_nodes(op="NMSRotated")[0]

        self.assertListEqual(ir_node.out_port(
            0).data.get_shape().tolist(), [None, 3])
        self.assertListEqual(ir_node.out_port(
            1).data.get_shape().tolist(), [None, 3])
        self.assertListEqual(ir_node.out_port(
            2).data.get_shape().tolist(), [1])

        self.assertEqual(ir_node["version"], "opset13")
        self.assertEqual(ir_node['sort_result_descending'], False)
        self.assertEqual(ir_node['output_type'], "i32")
        self.assertEqual(ir_node['clockwise'], False)
        self.assertEqual(loaded_model.get_output_element_type(0), Type.i32)
        self.assertEqual(loaded_model.get_output_element_type(1), Type.f32)
        self.assertEqual(loaded_model.get_output_element_type(2), Type.i32)

        self.assertEqual(loaded_model.get_output_partial_shape(
            0), PartialShape([Dimension(-1, 10), 3]))
        self.assertEqual(loaded_model.get_output_partial_shape(
            1), PartialShape([Dimension(-1, 10), 3]))
        self.assertEqual(loaded_model.get_output_partial_shape(
            2), PartialShape([1]))

    def test_nms_rotated_13_attrs_true_i64(self):
        boxes_shape = [1, 100, 5]
        scores_shape = [1, 3, 100]
        max_output_boxes_val = 5
        iou_threshold_val = 0.5
        score_threshold_val = 0.4

        boxes_parameter = opset13.parameter(
            boxes_shape, name="Boxes", dtype=np.float32)
        scores_parameter = opset13.parameter(
            scores_shape, name="Scores", dtype=np.float32)

        max_output_boxes = opset13.constant([max_output_boxes_val], np.int64)
        iou_threshold = opset13.constant([iou_threshold_val], np.float32)
        score_threshold = opset13.constant([score_threshold_val], np.float32)

        sort_result_descending = True
        output_type = "i64"
        clockwise = True

        node = opset13.nms_rotated(boxes_parameter, scores_parameter, max_output_boxes, iou_threshold,
                                   score_threshold, sort_result_descending, output_type, clockwise)

        model = Model(node, [boxes_parameter, scores_parameter])
        graph, loaded_model = TestOps.check_graph_can_save(
            model, 'nms_rotated_model_2')
        ir_node = graph.get_op_nodes(op="NMSRotated")[0]

        self.assertListEqual(ir_node.out_port(
            0).data.get_shape().tolist(), [None, 3])
        self.assertListEqual(ir_node.out_port(
            1).data.get_shape().tolist(), [None, 3])
        self.assertListEqual(ir_node.out_port(
            2).data.get_shape().tolist(), [1])

        self.assertEqual(ir_node["version"], "opset13")
        self.assertEqual(ir_node['sort_result_descending'], True)
        self.assertEqual(ir_node['output_type'], "i64")
        self.assertEqual(ir_node['clockwise'], True)
        self.assertEqual(loaded_model.get_output_element_type(0), Type.i64)
        self.assertEqual(loaded_model.get_output_element_type(1), Type.f32)
        self.assertEqual(loaded_model.get_output_element_type(2), Type.i64)

        self.assertEqual(loaded_model.get_output_partial_shape(
            0), PartialShape([Dimension(-1, 15), 3]))
        self.assertEqual(loaded_model.get_output_partial_shape(
            1), PartialShape([Dimension(-1, 15), 3]))
        self.assertEqual(loaded_model.get_output_partial_shape(
            2), PartialShape([1]))
