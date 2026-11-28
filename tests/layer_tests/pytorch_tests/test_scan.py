# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


def _scan(*args, **kwargs):
    from torch._higher_order_ops.scan import scan

    return scan(*args, **kwargs)


class TestScanFX(PytorchLayerTest):
    """Tests for the ``scan`` higher-order op from ``torch.export``.

    ``scan(combine_fn, init, xs, additional_inputs)`` runs ``combine_fn(carry, x)`` once per slice of
    ``xs`` along dim 0 and returns ``(final_carry, stacked_ys)``.
    """

    def _prepare_input(self):
        return tuple(self.random.randn(*shape) for shape in self.input_shapes)

    def create_model_single_carry(self):
        """The repro from huggingface/transformers#51306: one carry, one ``xs``, ``ys`` is the carry itself."""

        class ScanSingleCarry(torch.nn.Module):
            def forward(self, xs):
                def step(carry, x):
                    carry = carry * 0.5 + x
                    return carry, carry.clone()

                return _scan(step, torch.zeros_like(xs[0]), xs)

        return ScanSingleCarry(), "scan"

    def create_model_multiple_carries_and_xs(self):
        """Several carries of different ranks, several ``xs`` of different shapes, ``ys`` not a carry."""

        class ScanMulti(torch.nn.Module):
            def forward(self, a, b):
                def step(carry, x):
                    h, c = carry
                    xa, xb = x
                    h = h * torch.tanh(xa) + 1.0
                    c = c + xb.sum()
                    return (h, c), h * 2

                init = (torch.zeros_like(a[0]), torch.zeros(()))
                (h, c), ys = _scan(step, init, (a, b))
                return h, c, ys

        return ScanMulti(), "scan"

    def create_model_additional_inputs(self):
        """``combine_fn`` closes over a parameter, which export passes as an additional (loop-invariant) input."""

        class ScanAdditional(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, xs):
                def step(carry, x):
                    carry = torch.tanh(carry @ self.weight + x)
                    return carry, carry.clone()

                return _scan(step, torch.zeros_like(xs[0]), xs)

        return ScanAdditional(), "scan"

    def create_model_no_ys(self):
        """``combine_fn`` returns only the carry: the op has no stacked outputs."""

        class ScanNoYs(torch.nn.Module):
            def forward(self, xs):
                def step(carry, x):
                    return carry + x, []

                carry, _ = _scan(step, torch.zeros_like(xs[0]), xs)
                return carry

        return ScanNoYs(), "scan"

    def create_model_selective_scan(self):
        """The Mamba recurrence ``h = exp(dt * A) * h + dt * B * u`` over the sequence, returning every state."""

        class ScanSelective(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.A = torch.nn.Parameter(-torch.rand(3, 2))

            def forward(self, dt, bu):
                # dt, bu: [seq, batch, channels, state]
                def step(h, x):
                    dt_t, bu_t = x
                    h = torch.exp(dt_t * self.A) * h + dt_t * bu_t
                    return h, h.clone()

                return _scan(step, torch.zeros_like(bu[0]), (dt, bu))

        return ScanSelective(), "scan"

    def create_model_reverse_dim1(self):
        """``dim=1, reverse=True``: export lowers both to ``movedim``/``flip`` around the op, which walks dim 0."""

        class ScanReverseDim1(torch.nn.Module):
            def forward(self, xs):
                def step(carry, x):
                    carry = carry * 0.9 + x
                    return carry, carry.clone()

                return _scan(step, torch.zeros_like(xs[:, 0]), xs, dim=1, reverse=True)

        return ScanReverseDim1(), "scan"

    def _run(self, model_and_kind, ie_device, precision, ir_version, **kwargs):
        self._test(*model_and_kind, ie_device, precision, ir_version,
                   trace_model=False, use_torch_export=True, fx_kind="scan", **kwargs)

    @pytest.mark.parametrize("seq_len", [1, 5])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_scan_single_carry(self, seq_len, ie_device, precision, ir_version):
        self.input_shapes = [(seq_len, 2, 4)]
        self._run(self.create_model_single_carry(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_scan_multiple_carries_and_xs(self, ie_device, precision, ir_version):
        self.input_shapes = [(5, 2, 4), (5, 3)]
        self._run(self.create_model_multiple_carries_and_xs(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_scan_additional_inputs(self, ie_device, precision, ir_version):
        self.input_shapes = [(6, 2, 4)]
        self._run(self.create_model_additional_inputs(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_scan_no_ys(self, ie_device, precision, ir_version):
        self.input_shapes = [(4, 3)]
        self._run(self.create_model_no_ys(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_scan_selective_scan(self, ie_device, precision, ir_version):
        self.input_shapes = [(7, 2, 3, 2), (7, 2, 3, 2)]
        self._run(self.create_model_selective_scan(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_scan_reverse_dim1(self, ie_device, precision, ir_version):
        self.input_shapes = [(2, 5, 4)]
        self._run(self.create_model_reverse_dim1(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_scan_dynamic_length(self, ie_device, precision, ir_version):
        """The trip count comes from a dynamic dim 0, so it must be computed from the input shape at runtime."""
        self.input_shapes = [(5, 2, 4)]
        self._run(self.create_model_single_carry(), ie_device, precision, ir_version,
                  dynamic_shapes_for_export={"xs": {0: torch.export.Dim("seq", min=2)}})
