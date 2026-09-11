# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_alias(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        y[:, 1, :, :] = 4.
        return y


class aten_alias_tensor(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        n, c, h, w = x.shape
        ones = torch.ones([2, h, w]).to(x.dtype)
        y[:, 1:, :, :] = ones
        return y


class aten_loop_alias(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        for i in range(2):
            y[:, i, :, :] = 4.
        return y


class TestAliases(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(1, 3, 224, 224),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_alias(self, ie_device, precision, ir_version):
        self._test(aten_alias(), ["aten::slice",
                                  "aten::select",
                                  "aten::copy_"],
                   ie_device, precision, ir_version,
                   fx_kind=["aten.clone.default", "aten.select.int", "aten.fill_.Tensor"])

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_alias_tensor(self, ie_device, precision, ir_version):
        self._test(aten_alias_tensor(), ["aten::slice",
                                         "aten::copy_"],
                   ie_device, precision, ir_version, freeze_model=False,
                   fx_kind=["aten.slice.Tensor", "aten.copy_.default"])

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_loop_alias(self, ie_device, precision, ir_version):
        self._test(aten_loop_alias(), ["aten::slice",
                                       "aten::select",
                                       "aten::copy_",
                                       "prim::Loop"],
                   ie_device, precision, ir_version, freeze_model=False,
                   fx_kind=["aten.clone.default", "aten.select.int", "aten.fill_.Tensor"])


class TestAliasSchema(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(1, 2, 3, 4).astype("float32"),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("operation", ["clone", "matmul", "conv2d", "relu", "sigmoid", "slice"])
    def test_tensor_in_list(self, operation, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            __constants__ = ["operation"]

            def __init__(self, operation):
                super().__init__()
                self.operation = operation

            def forward(self, tensor):
                if self.operation == "clone":
                    result = tensor.clone()
                elif self.operation == "matmul":
                    result = torch.matmul(tensor, tensor.transpose(-1, -2))
                elif self.operation == "conv2d":
                    result = torch.conv2d(tensor, torch.ones(2, 2, 1, 1))
                elif self.operation == "relu":
                    result = tensor.relu()
                elif self.operation == "sigmoid":
                    result = tensor.sigmoid()
                else:
                    result = tensor[:, :, :, :2]
                result.add_(1)
                # Container use puts even independent tensors in AliasDb's wildcard set.
                values = [tensor, result]
                return values[0], values[1]

        self._test(Model(operation), [f"aten::{operation}", "aten::add_", "prim::ListConstruct"],
                   ie_device, precision, ir_version, freeze_model=False,
                   fx_kind=[f"aten.{operation}", "aten.add_"])
