# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestMatMul(PytorchLayerTest):
    def _prepare_input(self, matrix1_shape=(2, 2), matrix2_shape=(2, 2)):
        import numpy as np
        return (np.random.randn(*matrix1_shape).astype(np.float32), np.random.randn(*matrix2_shape).astype(np.float32))

    def create_model(self, op_type="aten::mm"):
        import torch
        ops = {
            "aten::mm": torch.mm,
            "aten::bmm": torch.bmm,
            "aten::matmul": torch.matmul
        }

        class aten_mm(torch.nn.Module):
            def __init__(self, op):
                super(aten_mm, self).__init__()
                self.op = op

            def forward(self, m1, m2):
                return self.op(m1, m2)

        ref_net = None

        return aten_mm(ops[op_type]), ref_net, op_type

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {'matrix1_shape': (3, 3), 'matrix2_shape': (3, 3)},
        {'matrix1_shape': (2, 3), 'matrix2_shape': (3, 2)},
        {'matrix1_shape': (10, 5), 'matrix2_shape': (5, 1)},
        {'matrix1_shape': (1, 10), 'matrix2_shape': (10, 2)},
        {'matrix1_shape': (1, 10), 'matrix2_shape': (10, 1)},

    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_mm(self, kwargs_to_prepare_input, ie_device, precision, ir_version):
        self._test(*self.create_model('aten::mm'), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input)

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {'matrix1_shape': (10, 3, 3), 'matrix2_shape': (10, 3, 3)},
        {'matrix1_shape': (1, 2, 3), 'matrix2_shape': (1, 3, 2)},
        {'matrix1_shape': (2, 10, 5), 'matrix2_shape': (2, 5, 1)},
        {'matrix1_shape': (3, 1, 10), 'matrix2_shape': (3, 10, 2)},
        {'matrix1_shape': (4, 1, 10), 'matrix2_shape': (4, 10, 1)},

    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_bmm(self, kwargs_to_prepare_input, ie_device, precision, ir_version):
        self._test(*self.create_model('aten::bmm'), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input)

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {'matrix1_shape': (10, 3, 3), 'matrix2_shape': (10, 3, 3)},
        {'matrix1_shape': (1, 2, 3), 'matrix2_shape': (1, 3, 2)},
        {'matrix1_shape': (2, 10, 5), 'matrix2_shape': (2, 5, 1)},
        {'matrix1_shape': (3, 1, 10), 'matrix2_shape': (3, 10, 2)},
        {'matrix1_shape': (4, 1, 10), 'matrix2_shape': (4, 10, 1)},
        {'matrix1_shape': (3, 3), 'matrix2_shape': (3, 3)},
        {'matrix1_shape': (2, 3), 'matrix2_shape': (3, 2)},
        {'matrix1_shape': (10, 5), 'matrix2_shape': (5, 1)},
        {'matrix1_shape': (1, 10), 'matrix2_shape': (10, 2)},
        {'matrix1_shape': (1, 10), 'matrix2_shape': (10, 1)},
        {'matrix1_shape': (10, 3, 3), 'matrix2_shape': (3, 3)},
        {'matrix1_shape': (2, 3), 'matrix2_shape': (10, 3, 2)},
        {'matrix1_shape': (1, 10, 5), 'matrix2_shape': (5, 1)},
        {'matrix1_shape': (5, 1, 10), 'matrix2_shape': (10, 2)},
        {'matrix1_shape': (1, 10), 'matrix2_shape': (4, 10, 2)},
        {'matrix1_shape': (2, 1, 10), 'matrix2_shape': (10, 1)},
        {'matrix1_shape': (1, 10), 'matrix2_shape': (2, 10, 1)},

    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_matmul(self, kwargs_to_prepare_input, ie_device, precision, ir_version):
        self._test(*self.create_model('aten::matmul'), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input)
