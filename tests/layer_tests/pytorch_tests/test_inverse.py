# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_inverse(torch.nn.Module):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, input_tensor):
        return torch.inverse(input_tensor)

class aten_inverse_out(torch.nn.Module):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, input_tensor, out):
        return torch.inverse(input_tensor, out=out), out

class TestInverse(PytorchLayerTest):
    def _prepare_input(self, out = False):
        if not out:
            return (self.input_tensor,)
        return (self.input_tensor, np.zeros_like(self.input_tensor))

    @pytest.mark.parametrize("data", [
        [
            [0.5, 1],
            [3, 2]
        ],
        [
            [
                [2, -1, 0],
                [-1, 2, -1],
                [0, -1, 2],
            ],
            [
                [3, 1, 2],
                [0, 4, 1],
                [2, -2, 0],
            ]
        ],
        [
            [7, -2, 5, 8],
            [-6, 3, -2, 27],
            [10, -12, 23, 21],
            [1, -21, 16, 15]
        ],
        [
            [5, 6, 6, 8],
            [2, 2, 2, 8],
            [6, 6, 2, 8],
            [2, 3, 6, 7]
        ]
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
    def test_inverse(self, data, dtype, out, ie_device, precision, ir_version):
        self.input_tensor = np.array(data, dtype=dtype)
        if ie_device == "GPU":
            pytest.xfail(reason="Inverse-14 is not supported on GPU")
        if not out:
            self._test(aten_inverse(), None, "aten::linalg_inv",
                    ie_device, precision, ir_version, trace_model=True, freeze_model=False)
        else:
            self._test(aten_inverse_out(), None, "aten::linalg_inv",
                    ie_device, precision, ir_version, trace_model=True, freeze_model=False, kwargs_to_prepare_input={"out": out})

    @pytest.mark.parametrize("shape", [
        (10, 2, 2),
        (3, 5, 5),
        (10, 10),
        (7, 6, 5, 4, 4)
    ])
    @pytest.mark.parametrize("dtype", [
        np.float64,
        np.float32
    ])
    @pytest.mark.parametrize("seed", [
        1, 2, 3
    ])
    @pytest.mark.parametrize("out", [
        False, 
        True
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_inverse(self, shape, dtype, seed, out, ie_device, precision, ir_version):
        rng = np.random.default_rng(seed)
        self.input_tensor = rng.uniform(-10.0, 10.0, shape).astype(dtype)
        if not out:
            self._test(aten_inverse(), None, "aten::linalg_inv",
                    ie_device, precision, ir_version, trace_model=True, freeze_model=False)
        else:
            self._test(aten_inverse_out(), None, "aten::linalg_inv",
                    ie_device, precision, ir_version, trace_model=True, freeze_model=False, kwargs_to_prepare_input={"out": out})
