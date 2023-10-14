# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import pytest

import openvino.tools.mo.front.onnx.activation_ext as extractors
from openvino.tools.mo.ops.activation_ops import Elu
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import PB
from unit_tests.utils.graph import build_graph


class TestActivationOpsONNXExtractorTest():
    @staticmethod
    def _create_node(op_name: str):
        pb = onnx.helper.make_node(op_name, ["X"], ["Y"])
        graph = build_graph({'node_0': {'pb': pb}}, [])
        return Node(graph, 'node_0')

    @staticmethod
    def _base_attrs(op_name: str):
        # reference output Node attributes
        return (
            dict(
                op=op_name,
            )
        )

    def _match(self, out, ref):
        for key in ref.keys():
            status = out[key] == ref[key]
            if type(status) in [list, np.ndarray]:
                status = np.all(status)
            assert status, f"Mismatch for field {key}, observed: {out[key]}, expected: {ref[key]}"

    @staticmethod
    def _extract(op_name):
        node = __class__._create_node(op_name)
        getattr(extractors, op_name + 'Extractor').extract(node)
        return node.graph.node[node.id]

    @pytest.mark.parametrize("op_name",['Abs', 'Acos', 'Asin', 'Atan', 'Acosh', 'Asinh', 'Atanh', 'Cos', 'Cosh', 'Erf', 'Exp', 'Floor', 'Log', 'Not', 'Sigmoid', 'Sin',
                'Sinh', 'Tan', 'Tanh'])
    def test_default(self, op_name):
        ref = self._base_attrs(op_name)
        if ref['op'] == 'Not':
            ref['op'] = 'LogicalNot'
        out = self._extract(op_name)
        self._match(out, ref)


class TestEluONNXExt():
    @staticmethod
    def _create_elu_node(alpha=1.0):
        pb = onnx.helper.make_node(
            'Elu',
            inputs=['x'],
            outputs=['y'],
            alpha=alpha
        )
        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Elu'] = Elu

    @pytest.mark.parametrize("alpha",[1.0, 2.0, 3.0])
    def test_elu_ext(self, alpha):
        node = self._create_elu_node(alpha)
        extractors.EluExtractor.extract(node)

        exp_res = {
            'type': 'Elu',
            'alpha': alpha,
            'infer': Elu.infer
        }

        for key in exp_res.keys():
            assert node[key] == exp_res[key]
