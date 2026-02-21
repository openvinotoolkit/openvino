# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestNanToNum(PytorchLayerTest):
    def _prepare_input(self, dtype, shape):
        # Create input with NaN, +Inf, -Inf, and normal values
        base = np.array([np.nan, np.inf, -np.inf, 1.25, -2.5, 0.0], dtype=dtype)
        if shape == (6,):
            x = base
        else:
            x = np.tile(base, (shape[0], 1))
        return (x,)

    def create_model(self, nan=None, posinf=None, neginf=None):
        class aten_nan_to_num(torch.nn.Module):
            def __init__(self, nan, posinf, neginf):
                super().__init__()
                self.nan = nan
                self.posinf = posinf
                self.neginf = neginf

            def forward(self, x):
                return torch.nan_to_num(x, nan=self.nan, posinf=self.posinf, neginf=self.neginf)

        return aten_nan_to_num(nan, posinf, neginf), None, "aten::nan_to_num"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("shape", [(6,), (2, 6)])
    def test_nan_to_num_defaults(self, dtype, shape, ie_device, precision, ir_version):
        # Test with default replacement values (nan=0, posinf=dtype max, neginf=dtype min)
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"dtype": dtype, "shape": shape},
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("nan", [0.0, 3.0])
    @pytest.mark.parametrize("posinf", [None, 100.0])
    @pytest.mark.parametrize("neginf", [None, -100.0])
    def test_nan_to_num_custom(self, dtype, nan, posinf, neginf, ie_device, precision, ir_version):
        self._test(
            *self.create_model(nan=nan, posinf=posinf, neginf=neginf),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"dtype": dtype, "shape": (6,)},
        )
