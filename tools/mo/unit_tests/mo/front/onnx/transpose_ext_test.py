# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import pytest

import numpy as np
import onnx

from openvino.tools.mo.front.onnx.transpose_ext import TransposeFrontExtractor
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import PB


class TestTransposeONNXExt():
    @staticmethod
    def _create_transpose_node(order: list):
        if order is None:
            # Default transpose
            pb = onnx.helper.make_node(
                'Transpose',
                inputs=['data'],
                outputs=['transposed'],
            )
        else:
            # Transpose with order
            pb = onnx.helper.make_node(
                'Transpose',
                inputs=['data'],
                outputs=['transposed'],
                perm=order
            )
        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Transpose'] = Transpose
        pass

    # This generator generates all permutations for [0,1,2,3] and [0,1,2] orders
    @pytest.mark.parametrize("order",[list(order) for order in list(itertools.permutations(np.arange(4)))] +
               [list(order) for order in list(itertools.permutations(np.arange(3)))] + [None])
    def test_transpose_ext(self, order):
        node = self._create_transpose_node(order)
        TransposeFrontExtractor.extract(node)

        exp_res = {
            'type': 'Transpose',
            'order': order,
            'infer': Transpose.infer
        }

        for key in exp_res.keys():
            if isinstance(exp_res[key], list):
                assert np.array_equal(node[key], exp_res[key]),\
                    "Orders are not the same: {} and {}".format(node[key], exp_res[key])
            else:
                assert node[key] == exp_res[key]
