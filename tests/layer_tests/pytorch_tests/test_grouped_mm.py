# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import openvino as ov

from pytorch_layer_test_class import PytorchLayerTest


class TestGroupedMMConstWeights(PytorchLayerTest):
    """`torch._grouped_mm` with weights baked into the module as a buffer.

    With the weights folded as a Constant in the IR, the CPU
    `ConvertGroupedMatMulToGatherMatmul` pass can lower the op to the internal
    `GatherMatmul` and the kernel `GatherMatmul::execute` should fire.
    """

    def _prepare_input(self, a_shape=(2, 3, 8)):
        return (self.random.randn(*a_shape).astype("float32"),)

    def create_model(self, b_shape=(2, 8, 16), bf16=False):
        import torch
        b_np = self.random.randn(*b_shape).astype("float32")

        class aten_grouped_mm_const_b(torch.nn.Module):
            def __init__(self, b, bf16):
                super().__init__()
                self.bf16 = bf16
                self.register_buffer("b", torch.from_numpy(b).bfloat16() if bf16 else torch.from_numpy(b))

            def forward(self, a):
                if self.bf16:
                    return torch._grouped_mm(a.to(torch.bfloat16), self.b).to(torch.float32)
                return torch._grouped_mm(a, self.b)

        return aten_grouped_mm_const_b(b_np, bf16), "aten::_grouped_mm"

    @pytest.mark.parametrize("a_shape,b_shape", [
        ((2, 3, 8), (2, 8, 16)),
        ((4, 1, 8), (4, 8, 8)),
        ((1, 8, 16), (1, 16, 8)),
        ((3, 7, 16), (3, 16, 8)),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_grouped_mm_const_b(self, a_shape, b_shape, ie_device, precision, ir_version):
        if ie_device.startswith("GPU"):
            if self.use_torch_export():
                pytest.skip("skip GPU BF16 torch export (FX) grouped_mm with no offsets")
            caps = ov.Core().get_property(ie_device, ov.properties.device.capabilities)
            if "GPU_HW_MATMUL" not in caps:
                pytest.skip("not supported on GPU without GPU_HW_MATMUL (immad)")
            if precision == "FP32":
                pytest.skip("GPU gather_matmul kernel does not support FP32")
        self._test(*self.create_model(b_shape=b_shape, bf16=self.use_torch_export()), ie_device, precision, ir_version,
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

    def create_model(self, k=8, n=8, num_groups=3, bf16=False):
        import torch
        b_np = self.random.randn(num_groups, k, n).astype("float32")

        class aten_grouped_mm_offs_const_b(torch.nn.Module):
            def __init__(self, b, bf16):
                super().__init__()
                self.bf16 = bf16
                self.register_buffer("b", torch.from_numpy(b).bfloat16() if bf16 else torch.from_numpy(b))

            def forward(self, a, offs):
                if self.bf16:
                    return torch._grouped_mm(a.to(torch.bfloat16), self.b, offs=offs).to(torch.float32)
                return torch._grouped_mm(a, self.b, offs=offs)

        return aten_grouped_mm_offs_const_b(b_np, bf16), "aten::_grouped_mm"

    @pytest.mark.parametrize("total_tokens,k,n,offsets", [
        (24, 8, 8, (8, 16, 24)),
        (16, 16, 8, (8, 16)),
        (8, 8, 16, (8,)),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_grouped_mm_offs_const_b(self, total_tokens, k, n, offsets, ie_device, precision, ir_version):
        if ie_device.startswith("GPU"):
            caps = ov.Core().get_property(ie_device, ov.properties.device.capabilities)
            if "GPU_HW_MATMUL" not in caps:
                pytest.skip("not supported on GPU without GPU_HW_MATMUL (immad)")
            if precision == "FP32":
                pytest.skip("GPU gather_matmul kernel does not support FP32")
        self._test(*self.create_model(k=k, n=n, num_groups=len(offsets), bf16=self.use_torch_export()),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"total_tokens": total_tokens, "offsets": offsets, "k": k},
                   trace_model=True)


# ---------------------------------------------------------------------------
# torch.nn.functional.grouped_mm  (public API, delegates to aten::_grouped_mm)
# ---------------------------------------------------------------------------

class TestFunctionalGroupedMMConstWeights(PytorchLayerTest):
    """`torch.nn.functional.grouped_mm` with weights baked in as a buffer (constant B)."""

    def _prepare_input(self, a_shape=(2, 3, 8)):
        return (self.random.randn(*a_shape).astype("float32"),)

    def create_model(self, b_shape=(2, 8, 16), bf16=False):
        import torch
        b_np = self.random.randn(*b_shape).astype("float32")

        class functional_grouped_mm_const_b(torch.nn.Module):
            def __init__(self, b, bf16):
                super().__init__()
                self.bf16 = bf16
                self.register_buffer("b", torch.from_numpy(b).bfloat16() if bf16 else torch.from_numpy(b))

            def forward(self, a):
                if self.bf16:
                    return torch.nn.functional.grouped_mm(a.to(torch.bfloat16), self.b).to(torch.float32)
                return torch.nn.functional.grouped_mm(a, self.b)

        return functional_grouped_mm_const_b(b_np, bf16), "aten::_grouped_mm"

    @pytest.mark.parametrize("a_shape,b_shape", [
        ((2, 3, 8),  (2, 8, 16)),
        ((4, 1, 8),  (4, 8, 8)),
        ((1, 8, 16), (1, 16, 8)),
        ((3, 7, 16), (3, 16, 8)),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_functional_grouped_mm_const_b(self, a_shape, b_shape, ie_device, precision, ir_version):
        if ie_device.startswith("GPU"):
            if self.use_torch_export():
                pytest.skip("skip GPU BF16 torch export (FX) grouped_mm with no offsets")
            caps = ov.Core().get_property(ie_device, ov.properties.device.capabilities)
            if "GPU_HW_MATMUL" not in caps:
                pytest.skip("not supported on GPU without GPU_HW_MATMUL (immad)")
            if precision == "FP32":
                pytest.skip("GPU gather_matmul kernel does not support FP32")
        self._test(*self.create_model(b_shape=b_shape, bf16=self.use_torch_export()), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"a_shape": a_shape},
                   trace_model=True)


class TestFunctionalGroupedMMOffsetsConstWeights(PytorchLayerTest):
    """`torch.nn.functional.grouped_mm` with `offs` and weights baked in as a buffer."""

    def _prepare_input(self, total_tokens=24, offsets=(8, 16, 24), k=8):
        import numpy as np
        return (
            self.random.randn(total_tokens, k).astype("float32"),
            np.asarray(offsets, dtype=np.int32),
        )

    def create_model(self, k=8, n=8, num_groups=3, bf16=False):
        import torch
        b_np = self.random.randn(num_groups, k, n).astype("float32")

        class functional_grouped_mm_offs_const_b(torch.nn.Module):
            def __init__(self, b, bf16):
                super().__init__()
                self.bf16 = bf16
                self.register_buffer("b", torch.from_numpy(b).bfloat16() if bf16 else torch.from_numpy(b))

            def forward(self, a, offs):
                if self.bf16:
                    return torch.nn.functional.grouped_mm(a.to(torch.bfloat16), self.b, offs=offs).to(torch.float32)
                return torch.nn.functional.grouped_mm(a, self.b, offs=offs)

        return functional_grouped_mm_offs_const_b(b_np, bf16), "aten::_grouped_mm"

    @pytest.mark.parametrize("total_tokens,k,n,offsets", [
        (24, 8,  8,  (8, 16, 24)),
        (16, 16, 8,  (8, 16)),
        (8,  8,  16, (8,)),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_functional_grouped_mm_offs_const_b(self, total_tokens, k, n, offsets, ie_device, precision, ir_version):
        if ie_device.startswith("GPU"):
            caps = ov.Core().get_property(ie_device, ov.properties.device.capabilities)
            if "GPU_HW_MATMUL" not in caps:
                pytest.skip("not supported on GPU without GPU_HW_MATMUL (immad)")
            if precision == "FP32":
                pytest.skip("GPU gather_matmul kernel does not support FP32")
        self._test(*self.create_model(k=k, n=n, num_groups=len(offsets), bf16=self.use_torch_export()),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"total_tokens": total_tokens, "offsets": offsets, "k": k},
                   trace_model=True)
