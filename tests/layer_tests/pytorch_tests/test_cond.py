# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestCondFX(PytorchLayerTest):
    """Test torch.cond operation for FX export mode."""

    def _prepare_input(self):
        rng = np.random.default_rng(43)
        return (rng.normal(size=self.input_shape).astype(np.float32), self.pred_value)

    def create_simple_cond_model(self):
        import torch

        class SimpleCond(torch.nn.Module):
            def forward(self, x, pred):
                def true_fn(x):
                    return x * 2

                def false_fn(x):
                    return x + 1

                return torch.cond(pred, true_fn, false_fn, (x,))

        return SimpleCond(), None, "cond"

    def create_multi_output_cond_model(self):
        import torch

        class MultiOutputCond(torch.nn.Module):
            def forward(self, x, pred):
                def true_fn(x):
                    return x * 2, x + 1

                def false_fn(x):
                    return x + 1, x * 2

                return torch.cond(pred, true_fn, false_fn, (x,))

        return MultiOutputCond(), None, "cond"

    def create_linear_cond_model(self, hidden_size):
        import torch

        class LinearCond(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.proj = torch.nn.Linear(hidden_size, hidden_size)

            def forward(self, x, is_prefill):
                def prefill_fn(x):
                    return self.proj(x) * x.sum(dim=-1, keepdim=True)

                def decode_fn(x):
                    return self.proj(x)

                return torch.cond(is_prefill, prefill_fn, decode_fn, (x,))

        return LinearCond(hidden_size), None, "cond"

    @pytest.mark.parametrize("pred_value", [np.array(True), np.array(False)])
    @pytest.mark.parametrize("input_shape", [(1, 4), (2, 3, 4)])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_simple_cond(self, pred_value, input_shape, ie_device, precision, ir_version):
        self.pred_value = pred_value
        self.input_shape = input_shape
        self._test(*self.create_simple_cond_model(), ie_device, precision, ir_version,
                   use_convert_model=True, fx_kind="cond")

    @pytest.mark.parametrize("pred_value", [np.array(True), np.array(False)])
    @pytest.mark.parametrize("input_shape", [(1, 4)])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_multi_output_cond(self, pred_value, input_shape, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("Multi-output If not supported on GPU - CVS-178157")
        self.pred_value = pred_value
        self.input_shape = input_shape
        self._test(*self.create_multi_output_cond_model(), ie_device, precision, ir_version,
                   use_convert_model=True, fx_kind="cond")

    @pytest.mark.parametrize("pred_value", [np.array(True), np.array(False)])
    @pytest.mark.parametrize("input_shape,hidden_size", [((1, 10, 64), 64)])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_linear_cond(self, pred_value, input_shape, hidden_size, ie_device, precision, ir_version):
        self.pred_value = pred_value
        self.input_shape = input_shape
        self._test(*self.create_linear_cond_model(hidden_size), ie_device, precision, ir_version,
                   use_convert_model=True, fx_kind="cond")
