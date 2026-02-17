# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestTupleUnpack(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 4, 6, 8),)

    def create_model(self):

        import torch

        class TupleArgument(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dims = (1, 2, 3, 4)

            def forward(self, input_data):
                N, _, H, W = self.dims
                return input_data * N * H * W

        return TupleArgument(), "prim::TupleUnpack"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_touple_unpack(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=False, freeze_model=False)


class TestComplexTupleUnpack(PytorchLayerTest):
    """Test TupleUnpack with complex tensors (ComplexTypeMark preservation).

    This tests the changes in:
    - src/frontends/pytorch/src/op/tuple_unpack.cpp
    - src/frontends/pytorch/src/transforms/tuple_unpack_replacer.cpp

    The test verifies that complex tensor operations work correctly when
    tensors are used in tuple-like patterns. We use trace_model=True to
    get proper TorchScript representation.
    """

    def _prepare_input(self):
        return (self.random.randn(2, 4, 2),)

    def create_model(self):
        class ComplexTupleUnpack(torch.nn.Module):
            def __init__(self, rng):
                super().__init__()
                # Use separate buffers instead of tuple
                freqs = rng.torch_randn(4, 2)
                complex_freqs = torch.view_as_complex(freqs)
                self.register_buffer('freqs_a', torch.view_as_real(complex_freqs))
                self.register_buffer('freqs_b', torch.view_as_real(complex_freqs * 2))

            def forward(self, x):
                complex_x = torch.view_as_complex(x)
                # Use view_as_complex on buffers
                a = torch.view_as_complex(self.freqs_a)
                b = torch.view_as_complex(self.freqs_b)
                result = complex_x * a + complex_x * b
                return torch.view_as_real(result)

        return ComplexTupleUnpack(self.random), "aten::mul"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_complex_tuple_unpack(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=True)


class TestComplexTupleUnpackMultiple(PytorchLayerTest):
    """Test TupleUnpack with multiple complex tensor outputs."""

    def _prepare_input(self):
        return (self.random.randn(2, 4, 2),)

    def create_model(self):
        class ComplexTupleUnpackMultiple(torch.nn.Module):
            def __init__(self, rng):
                super().__init__()
                freqs = rng.torch_randn(4, 2)
                complex_freqs = torch.view_as_complex(freqs)
                self.register_buffer('freqs_a', torch.view_as_real(complex_freqs))
                self.register_buffer('freqs_b', torch.view_as_real(complex_freqs * 2))
                self.register_buffer('freqs_c', torch.view_as_real(complex_freqs * 3))

            def forward(self, x):
                complex_x = torch.view_as_complex(x)
                a = torch.view_as_complex(self.freqs_a)
                b = torch.view_as_complex(self.freqs_b)
                c = torch.view_as_complex(self.freqs_c)
                result = complex_x * a + complex_x * b + complex_x * c
                return torch.view_as_real(result)

        return ComplexTupleUnpackMultiple(self.random), "aten::mul"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_complex_tuple_unpack_multiple(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=True)
