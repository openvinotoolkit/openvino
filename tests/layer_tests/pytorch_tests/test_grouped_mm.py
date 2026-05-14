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
        # GPU lowers GroupedMatMul only when B is a Constant (gather_matmul kernel needs constant weights).
        if ie_device.startswith("GPU"):
            pytest.skip("GPU requires constant weights for GroupedMatMul lowering")
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
        # GPU lowers GroupedMatMul only when B is a Constant (gather_matmul kernel needs constant weights).
        if ie_device.startswith("GPU"):
            pytest.skip("GPU requires constant weights for GroupedMatMul lowering")
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input,
                   trace_model=True)


class TestGroupedMMConstWeights(PytorchLayerTest):
    """`torch._grouped_mm` with weights baked into the module as a buffer.

    With the weights folded as a Constant in the IR, the CPU
    `ConvertGroupedMatMulToGatherMatmul` pass can lower the op to the internal
    `GatherMatmul` and the kernel `GatherMatmul::execute` should fire.
    """

    def _prepare_input(self, a_shape=(2, 3, 8)):
        return (self.random.randn(*a_shape).astype("float32"),)

    def create_model(self, b_shape=(2, 8, 16)):
        import torch
        b_np = self.random.randn(*b_shape).astype("float32")

        class aten_grouped_mm_const_b(torch.nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("b", torch.from_numpy(b))

            def forward(self, a):
                return torch._grouped_mm(a, self.b)

        return aten_grouped_mm_const_b(b_np), "aten::_grouped_mm"

    @pytest.mark.parametrize("a_shape,b_shape", [
        ((2, 3, 8), (2, 8, 16)),
        ((4, 1, 8), (4, 8, 8)),
        ((1, 8, 16), (1, 16, 8)),
        ((3, 7, 16), (3, 16, 8)),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_grouped_mm_const_b(self, a_shape, b_shape, ie_device, precision, ir_version):
        # GPU `gather_matmul` kernel is f16-only (DPAS).
        if ie_device.startswith("GPU") and precision == "FP32":
            pytest.skip("GPU gather_matmul kernel does not support f32 activations/weights")
        self._test(*self.create_model(b_shape=b_shape), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"a_shape": a_shape},
                   trace_model=True)


class TestGroupedMMOffsetsConstWeights(PytorchLayerTest):
    """`torch._grouped_mm` with `offs` and weights baked in as a buffer."""

    def _prepare_input(self, total_tokens=24, offsets=(8, 16, 24), k=8):
        import numpy as np
        return (
            self.random.randn(total_tokens, k).astype("float32"),
            np.asarray(offsets, dtype=np.int32),
        )

    def create_model(self, k=8, n=8, num_groups=3):
        import torch
        b_np = self.random.randn(num_groups, k, n).astype("float32")

        class aten_grouped_mm_offs_const_b(torch.nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("b", torch.from_numpy(b))

            def forward(self, a, offs):
                return torch._grouped_mm(a, self.b, offs=offs)

        return aten_grouped_mm_offs_const_b(b_np), "aten::_grouped_mm"

    @pytest.mark.parametrize("total_tokens,k,n,offsets", [
        (24, 8, 8, (8, 16, 24)),
        (16, 16, 8, (8, 16)),
        (8, 8, 16, (8,)),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_grouped_mm_offs_const_b(self, total_tokens, k, n, offsets, ie_device, precision, ir_version):
        # GPU `gather_matmul` kernel is f16-only (DPAS).
        if ie_device.startswith("GPU") and precision == "FP32":
            pytest.skip("GPU gather_matmul kernel does not support f32 activations/weights")
        self._test(*self.create_model(k=k, n=n, num_groups=len(offsets)),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"total_tokens": total_tokens, "offsets": offsets, "k": k},
                   trace_model=True)
