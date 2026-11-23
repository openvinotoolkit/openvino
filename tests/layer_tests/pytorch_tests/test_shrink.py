# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from pytorch_layer_test_class import PytorchLayerTest


class aten_hardshrink(torch.nn.Module):
    def __init__(self, lambd, dtype):
        super().__init__()
        self.lambd = lambd
        self.dtype = dtype
        if lambd is None:
            self.forward = self.forward_no_lambd

    def forward(self, x):
        x_copy = x.to(self.dtype)
        return F.hardshrink(x_copy, lambd=self.lambd)

    def forward_no_lambd(self, x):
        x_copy = x.to(self.dtype)
        return F.hardshrink(x_copy)


class aten_softshrink(torch.nn.Module):
    def __init__(self, lambd, dtype):
        super().__init__()
        self.lambd = lambd
        self.dtype = dtype
        if lambd is None:
            self.forward = self.forward_no_lambd

    def forward(self, x):
        x_copy = x.to(self.dtype)
        return F.softshrink(x_copy, lambd=self.lambd)

    def forward_no_lambd(self, x):
        x_copy = x.to(self.dtype)
        return F.softshrink(x_copy)


class TestHardshrink(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 4, 16, 16),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("lambd", [None, 0.5, 1.2, 0.0])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    def test_hardshrink(self, lambd, dtype, ie_device, precision, ir_version):
        kwargs = {}
        if dtype == torch.float16:
            kwargs["custom_eps"] = 1e-2
        self._test(
            aten_hardshrink(lambd, dtype),
            "aten::hardshrink",
            ie_device,
            precision,
            ir_version,
            **kwargs,
        )


class TestHardshrinkBoundary(PytorchLayerTest):
    def _prepare_input(self):
        # Explicit values crossing boundaries: -2.0, -0.5, -0.2, 0.0, 0.2, 0.5, 2.0
        return (
            torch.tensor(
                [[-2.0, -0.5, -0.2], [0.0, 0.2, 0.5], [1.0, 2.0, -1.0]],
                dtype=torch.float32,
            ),
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("lambd", [0.5, 1.0])
    def test_hardshrink_boundary(self, lambd, ie_device, precision, ir_version):
        self._test(
            aten_hardshrink(lambd, torch.float32),
            "aten::hardshrink",
            ie_device,
            precision,
            ir_version,
        )


class TestHardshrinkAdversarial(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "input_tensor",
        [
            torch.tensor(0.75, dtype=torch.float32),  # Rank-0 scalar
            torch.linspace(-2.0, 2.0, 21, dtype=torch.float32),  # 1D
            torch.randn(2, 3, 4, 5, 6, dtype=torch.float32),  # 5D High-rank
            torch.empty((0, 4), dtype=torch.float32),  # Empty tensor
        ],
    )
    @pytest.mark.parametrize("lambd", [0.5, 1.25])
    def test_hardshrink_adversarial(
        self, input_tensor, lambd, ie_device, precision, ir_version
    ):
        self.input_tensor = input_tensor
        self._test(
            aten_hardshrink(lambd, torch.float32),
            "aten::hardshrink",
            ie_device,
            precision,
            ir_version,
        )


class TestHardshrinkSpecialFloats(PytorchLayerTest):
    def _prepare_input(self):
        return (
            torch.tensor(
                [
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                    0.0,
                    -0.0,
                    0.5,
                    -0.5,
                    0.5001,
                    -0.5001,
                ],
                dtype=torch.float32,
            ),
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("lambd", [0.5, 0.0])
    def test_hardshrink_special_floats(self, lambd, ie_device, precision, ir_version):
        self._test(
            aten_hardshrink(lambd, torch.float32),
            "aten::hardshrink",
            ie_device,
            precision,
            ir_version,
        )


class TestSoftshrink(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 4, 16, 16),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("lambd", [None, 0.5, 1.2, 0.0])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    def test_softshrink(self, lambd, dtype, ie_device, precision, ir_version):
        kwargs = {}
        if dtype == torch.float16:
            kwargs["custom_eps"] = 1e-2
        self._test(
            aten_softshrink(lambd, dtype),
            "aten::softshrink",
            ie_device,
            precision,
            ir_version,
            **kwargs,
        )


class TestSoftshrinkBoundary(PytorchLayerTest):
    def _prepare_input(self):
        # Explicit values crossing boundaries: -2.0, -0.5, -0.2, 0.0, 0.2, 0.5, 2.0
        return (
            torch.tensor(
                [[-2.0, -0.5, -0.2], [0.0, 0.2, 0.5], [1.0, 2.0, -1.0]],
                dtype=torch.float32,
            ),
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("lambd", [0.5, 1.0])
    def test_softshrink_boundary(self, lambd, ie_device, precision, ir_version):
        self._test(
            aten_softshrink(lambd, torch.float32),
            "aten::softshrink",
            ie_device,
            precision,
            ir_version,
        )


class TestSoftshrinkAdversarial(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "input_tensor",
        [
            torch.tensor(0.75, dtype=torch.float32),  # Rank-0 scalar
            torch.linspace(-2.0, 2.0, 21, dtype=torch.float32),  # 1D
            torch.randn(2, 3, 4, 5, 6, dtype=torch.float32),  # 5D High-rank
            torch.empty((0, 4), dtype=torch.float32),  # Empty tensor
        ],
    )
    @pytest.mark.parametrize("lambd", [0.5, 1.25])
    def test_softshrink_adversarial(
        self, input_tensor, lambd, ie_device, precision, ir_version
    ):
        self.input_tensor = input_tensor
        self._test(
            aten_softshrink(lambd, torch.float32),
            "aten::softshrink",
            ie_device,
            precision,
            ir_version,
        )


class TestSoftshrinkSpecialFloats(PytorchLayerTest):
    def _prepare_input(self):
        return (
            torch.tensor(
                [
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                    0.0,
                    -0.0,
                    0.5,
                    -0.5,
                    0.5001,
                    -0.5001,
                ],
                dtype=torch.float32,
            ),
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("lambd", [0.5, 0.0])
    def test_softshrink_special_floats(self, lambd, ie_device, precision, ir_version):
        self._test(
            aten_softshrink(lambd, torch.float32),
            "aten::softshrink",
            ie_device,
            precision,
            ir_version,
        )
