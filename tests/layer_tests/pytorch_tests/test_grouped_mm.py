# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestGroupedMM(PytorchLayerTest):
    """Tests torch._grouped_mm — the 3D x 3D no-`offs` (batched matmul) case."""

    def _prepare_input(self, a_shape=(2, 3, 8), b_shape=(2, 8, 16)):
        return (
            self.random.randn(*a_shape).astype("float32"),
            self.random.randn(*b_shape).astype("float32"),
        )

    def create_model(self):
        import torch

        class aten_grouped_mm(torch.nn.Module):
            def forward(self, a, b):
                return torch._grouped_mm(a, b)

        return aten_grouped_mm(), "aten::_grouped_mm"

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"a_shape": (2, 3, 8), "b_shape": (2, 8, 16)},
        {"a_shape": (4, 1, 8), "b_shape": (4, 8, 8)},
        {"a_shape": (1, 8, 16), "b_shape": (1, 16, 8)},
        {"a_shape": (3, 7, 16), "b_shape": (3, 16, 8)},
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_grouped_mm(self, kwargs_to_prepare_input, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("_grouped_mm is not supported on GPU")
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input,
                   trace_model=True)


class TestGroupedMMOffsets(PytorchLayerTest):
    """Tests torch._grouped_mm with `offs` (2D x 3D MoE case)."""

    def _prepare_input(self, total_tokens=10, k=4, n=5, offsets=(3, 7, 10)):
        import numpy as np
        return (
            self.random.randn(total_tokens, k).astype("float32"),
            self.random.randn(len(offsets), k, n).astype("float32"),
            np.asarray(offsets, dtype=np.int32),
        )

    def create_model(self):
        import torch

        class aten_grouped_mm_offs(torch.nn.Module):
            def forward(self, a, b, offs):
                return torch._grouped_mm(a, b, offs=offs)

        return aten_grouped_mm_offs(), "aten::_grouped_mm"

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"total_tokens": 24, "k": 8, "n": 8, "offsets": (8, 16, 24)},
        {"total_tokens": 16, "k": 16, "n": 8, "offsets": (8, 16)},
        {"total_tokens": 8, "k": 8, "n": 16, "offsets": (8,)},
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_grouped_mm_offs(self, kwargs_to_prepare_input, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("_grouped_mm is not supported on GPU")
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input,
                   trace_model=True)
