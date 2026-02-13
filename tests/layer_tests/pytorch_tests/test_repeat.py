# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestRepeat(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 10),)

    def create_model(self, repeats):
        import torch

        class aten_repeat(torch.nn.Module):
            def __init__(self, repeats):
                super().__init__()
                self.repeats = repeats

            def forward(self, x):
                return x.repeat(self.repeats)


        return aten_repeat(repeats), "aten::repeat"

    @pytest.mark.parametrize("repeats", [(4, 3), (1, 1), (1, 2, 3), (1, 2, 2, 3)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_repeat(self, repeats, ie_device, precision, ir_version):
        self._test(*self.create_model(repeats), ie_device, precision, ir_version)


class TestRepeatList(PytorchLayerTest):
    def _prepare_input(self, repeats_shape):
        return (self.random.randn(2, 10), self.random.randn(*repeats_shape),)

    def create_model(self):
        import torch

        class aten_repeat(torch.nn.Module):

            def forward(self, x, y):
                y_shape = y.shape
                return x.repeat([y_shape[0], y_shape[1]])


        return aten_repeat(), ["aten::repeat", "prim::ListConstruct"]

    @pytest.mark.parametrize("repeats", [(4, 3), (1, 1), (1, 3, 3), (1, 2, 2, 3)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_repeat(self, repeats, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"repeats_shape": repeats}, fx_kind="aten.repeat.default")


class TestRepeatFromFlanT5(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(1, 15),)

    def create_model(self):
        import torch
        from transformers.modeling_utils import ModuleUtilsMixin

        class aten_repeat(torch.nn.Module):
            def forward(self, x):
                return ModuleUtilsMixin.create_extended_attention_mask_for_decoder(x.size(), x)

        return aten_repeat(), "aten::repeat"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_repeat_t5(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, trace_model=True, use_convert_model=True)
