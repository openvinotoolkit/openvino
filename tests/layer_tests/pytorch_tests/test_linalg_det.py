# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_linalg_det(torch.nn.Module):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, input_tensor):
        return torch.linalg.det(input_tensor)


class aten_linalg_det_out(torch.nn.Module):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, input_tensor, out):
        return torch.linalg.det(input_tensor, out=out), out


class TestLinalgDet(PytorchLayerTest):
    def _prepare_input(self, out=False):
        if not out:
            return (self.input_tensor,)
        out_shape = self.input_tensor.shape[:-2] \
            if self.input_tensor.ndim > 2 else ()
        return (self.input_tensor,
                np.zeros(out_shape, dtype=self.input_tensor.dtype))

    @pytest.mark.parametrize("data", [
        # 1x1: det = element itself
        [[5.0]],
        # 2x2: det = ad - bc = 0.5*2 - 1*3 = -2.0
        [
            [0.5, 1.0],
            [3.0, 2.0]
        ],
        # 2x2 singular: det = 0
        [
            [1.0, 2.0],
            [2.0, 4.0]
        ],
        # 3x3 identity: det = 1
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        # 3x3 permutation matrix (odd permutation): det = -1
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        # 3x3 triangular: det = product of diagonal = 2*3*4 = 24
        [
            [2.0, 1.0, 5.0],
            [0.0, 3.0, 7.0],
            [0.0, 0.0, 4.0]
        ],
        # 4x4 identity: det = 1
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        # 4x4 diagonal: det = 2*3*4*5 = 120
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 5.0]
        ],
        # 4x4 general
        [
            [7.0, -2.0, 5.0, 8.0],
            [-6.0, 3.0, -2.0, 27.0],
            [10.0, -12.0, 23.0, 21.0],
            [1.0, -21.0, 16.0, 15.0]
        ],
    ])
    @pytest.mark.parametrize("dtype", [
        np.float64,
        np.float32
    ])
    @pytest.mark.parametrize("out", [
        False,
        True
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_linalg_det(self, data, dtype, out, ie_device, precision, ir_version):
        self.input_tensor = np.array(data, dtype=dtype)
        if not out:
            self._test(aten_linalg_det(), None, "aten::linalg_det",
                       ie_device, precision, ir_version,
                       trace_model=True, freeze_model=False,
                       use_convert_model=True,
                       dynamic_shapes=False)
        else:
            self._test(aten_linalg_det_out(), None, "aten::linalg_det",
                       ie_device, precision, ir_version,
                       trace_model=True, freeze_model=False,
                       use_convert_model=True,
                       dynamic_shapes=False,
                       kwargs_to_prepare_input={"out": out})

    @pytest.mark.parametrize("data", [
        # Batch of 2x2 matrices
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ],
        # Batch of 3x3 matrices
        [
            [
                [2.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 2.0]
            ],
            [
                [3.0, 1.0, 2.0],
                [0.0, 4.0, 1.0],
                [2.0, -2.0, 0.0]
            ]
        ],
    ])
    @pytest.mark.parametrize("dtype", [
        np.float64,
        np.float32
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_linalg_det_batched(self, data, dtype, ie_device, precision, ir_version):
        self.input_tensor = np.array(data, dtype=dtype)
        self._test(aten_linalg_det(), None, "aten::linalg_det",
                   ie_device, precision, ir_version,
                   trace_model=True, freeze_model=False,
                   use_convert_model=True,
                   dynamic_shapes=False)
