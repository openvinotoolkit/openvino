# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestGroupedMM(PytorchLayerTest):
    """Tests torch._grouped_mm — the 3D x 3D no-`offs` (batched matmul) case."""

    def _prepare_input(self, a_shape=(2, 3, 16), b_shape=(2, 16, 8)):
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
        {"a_shape": (2, 3, 16), "b_shape": (2, 16, 8)},
        {"a_shape": (4, 1, 8), "b_shape": (4, 8, 8)},
        {"a_shape": (1, 8, 8),  "b_shape": (1, 8, 16)},
        {"a_shape": (3, 7, 8),  "b_shape": (3, 8, 16)},
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
            self.random.randn(total_tokens, n).astype("float32"),  # A: [T, N] — contracts with B[-2]=N
            self.random.randn(len(offsets), n, k).astype("float32"),
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

    def create_model(self, b_shape=(2, 16, 8)):
        import torch
        b_np = self.random.randn(*b_shape).astype("float32")  # B: [G, N, K]

        class aten_grouped_mm_const_b(torch.nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("b", torch.from_numpy(b))

            def forward(self, a):
                return torch._grouped_mm(a, self.b)

        return aten_grouped_mm_const_b(b_np), "aten::_grouped_mm"

    @pytest.mark.parametrize("a_shape,b_shape", [
        ((2, 3, 16), (2, 16, 8)),
        ((4, 1, 8),  (4, 8, 8)),
        ((1, 8, 8),  (1, 8, 16)),
        ((3, 7, 8),  (3, 8, 16)),
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

    def _prepare_input(self, total_tokens=24, offsets=(8, 16, 24), k=8, n=8):
        import numpy as np
        return (
            self.random.randn(total_tokens, n).astype("float32"),  # A: [T, N] — contracts with B[-2]=N
            np.asarray(offsets, dtype=np.int32),
        )

    def create_model(self, k=8, n=8, num_groups=3):
        import torch
        b_np = self.random.randn(num_groups, n, k).astype("float32")

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
                   kwargs_to_prepare_input={"total_tokens": total_tokens, "offsets": offsets, "k": k, "n": n},
                   trace_model=True)


class TestGroupedMMBias(PytorchLayerTest):
    """3D x 3D with runtime bias — all tensors are runtime inputs."""

    def _prepare_input(self, a_shape=(2, 3, 16), b_shape=(2, 16, 8)):
        g, _, k = b_shape  # bias: [G, K] where K = output dim
        return (
            self.random.randn(*a_shape).astype("float32"),
            self.random.randn(*b_shape).astype("float32"),
            self.random.randn(g, k).astype("float32"),  # bias [G, K]
        )

    def create_model(self):
        import torch

        class aten_grouped_mm_bias(torch.nn.Module):
            def forward(self, a, b, bias):
                return torch._grouped_mm(a, b, bias=bias)

        return aten_grouped_mm_bias(), "aten::_grouped_mm"

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"a_shape": (2, 3, 16), "b_shape": (2, 16, 8)},
        {"a_shape": (4, 1, 8),  "b_shape": (4, 8, 8)},
        {"a_shape": (1, 8, 8),  "b_shape": (1, 8, 16)},
        {"a_shape": (3, 7, 8),  "b_shape": (3, 8, 16)},
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_grouped_mm_bias(self, kwargs_to_prepare_input, ie_device, precision, ir_version):
        if ie_device.startswith("GPU"):
            pytest.skip("GPU requires constant weights for GroupedMatMul lowering")
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input,
                   trace_model=True)


class TestGroupedMMOffsetsBias(PytorchLayerTest):
    """2D x 3D with `offs` and runtime bias — all tensors are runtime inputs."""

    def _prepare_input(self, total_tokens=24, k=8, n=8, offsets=(8, 16, 24)):
        import numpy as np
        g = len(offsets)
        return (
            self.random.randn(total_tokens, n).astype("float32"),  # A: [T, N]
            self.random.randn(g, n, k).astype("float32"),          # B: [G, N, K]
            np.asarray(offsets, dtype=np.int32),
            self.random.randn(g, k).astype("float32"),             # bias: [G, K]
        )

    def create_model(self):
        import torch

        class aten_grouped_mm_offs_bias(torch.nn.Module):
            def forward(self, a, b, offs, bias):
                return torch._grouped_mm(a, b, bias=bias, offs=offs)

        return aten_grouped_mm_offs_bias(), "aten::_grouped_mm"

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"total_tokens": 24, "k": 8,  "n": 8,  "offsets": (8, 16, 24)},
        {"total_tokens": 16, "k": 16, "n": 8,  "offsets": (8, 16)},
        {"total_tokens": 8,  "k": 8,  "n": 16, "offsets": (8,)},
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_grouped_mm_offs_bias(self, kwargs_to_prepare_input, ie_device, precision, ir_version):
        if ie_device.startswith("GPU"):
            pytest.skip("GPU requires constant weights for GroupedMatMul lowering")
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input,
                   trace_model=True)


class TestGroupedMMConstWeightsBias(PytorchLayerTest):
    """`torch._grouped_mm` 3D x 3D with weights and bias baked in as buffers.

    Exercises the bias branch of ConvertGroupedMatMulToGatherMatmul (Case 2,
    GatherMatmul output + Add(Unsqueeze(bias))).
    """

    def _prepare_input(self, a_shape=(2, 3, 8)):
        return (self.random.randn(*a_shape).astype("float32"),)

    def create_model(self, b_shape=(2, 16, 8)):
        import torch
        g, _, k = b_shape  # bias: [G, K]
        b_np    = self.random.randn(*b_shape).astype("float32")
        bias_np = self.random.randn(g, k).astype("float32")

        class aten_grouped_mm_const_b_bias(torch.nn.Module):
            def __init__(self, b, bias):
                super().__init__()
                self.register_buffer("b",    torch.from_numpy(b))
                self.register_buffer("bias", torch.from_numpy(bias))

            def forward(self, a):
                return torch._grouped_mm(a, self.b, bias=self.bias)

        return aten_grouped_mm_const_b_bias(b_np, bias_np), "aten::_grouped_mm"

    @pytest.mark.parametrize("a_shape,b_shape", [
        ((2, 3, 16), (2, 16, 8)),
        ((4, 1, 8),  (4, 8, 8)),
        ((1, 8, 8),  (1, 8, 16)),
        ((3, 7, 8),  (3, 8, 16)),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_grouped_mm_const_b_bias(self, a_shape, b_shape, ie_device, precision, ir_version):
        if ie_device.startswith("GPU") and precision == "FP32":
            pytest.skip("GPU gather_matmul kernel does not support f32 activations/weights")
        self._test(*self.create_model(b_shape=b_shape), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"a_shape": a_shape},
                   trace_model=True)


class TestGroupedMMOffsetsConstWeightsBias(PytorchLayerTest):
    """`torch._grouped_mm` with `offs`, weights and bias baked in as buffers.

    Exercises the bias branch of ConvertGroupedMatMulToGatherMatmul (Case 1,
    Squeeze(GatherMatmul(...)) + Add(Gather(bias, idx_1d))).
    """

    def _prepare_input(self, total_tokens=24, offsets=(8, 16, 24), k=8, n=8):
        import numpy as np
        return (
            self.random.randn(total_tokens, n).astype("float32"),  # A: [T, N]
            np.asarray(offsets, dtype=np.int32),
        )

    def create_model(self, k=8, n=8, num_groups=3):
        import torch
        b_np    = self.random.randn(num_groups, n, k).astype("float32")
        bias_np = self.random.randn(num_groups, k).astype("float32")  # bias: [G, K]

        class aten_grouped_mm_offs_const_b_bias(torch.nn.Module):
            def __init__(self, b, bias):
                super().__init__()
                self.register_buffer("b",    torch.from_numpy(b))
                self.register_buffer("bias", torch.from_numpy(bias))

            def forward(self, a, offs):
                return torch._grouped_mm(a, self.b, bias=self.bias, offs=offs)

        return aten_grouped_mm_offs_const_b_bias(b_np, bias_np), "aten::_grouped_mm"

    @pytest.mark.parametrize("total_tokens,k,n,offsets", [
        (24, 8, 8,  (8, 16, 24)),
        (16, 16, 8, (8, 16)),
        (8,  8, 16, (8,)),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_grouped_mm_offs_const_b_bias(self, total_tokens, k, n, offsets, ie_device, precision, ir_version):
        if ie_device.startswith("GPU") and precision == "FP32":
            pytest.skip("GPU gather_matmul kernel does not support f32 activations/weights")
        self._test(*self.create_model(k=k, n=n, num_groups=len(offsets)),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"total_tokens": total_tokens, "offsets": offsets, "k": k, "n": n},
                   trace_model=True)
