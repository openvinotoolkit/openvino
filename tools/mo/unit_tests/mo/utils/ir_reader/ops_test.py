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
        graph = TestOps.check_graph_can_save(model, 'if_model')
        if_node = graph.get_op_nodes(op="If")[0]
        self.assertEqual(if_node["version"], "opset8")
        _, layer_info, _ = if_node['IE'][0]
        _, callable_attribute = layer_info[0]
        if callable(callable_attribute):
            self.assertEqual(callable_attribute(if_node), "If_opset8")
