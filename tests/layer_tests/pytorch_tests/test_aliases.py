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


@pytest.mark.nightly
@pytest.mark.precommit
@pytest.mark.precommit_torch_export
class TestViewMutations(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(3, 4, 5).astype("float32"),)

    @pytest.mark.parametrize("view", ["view", "transpose", "permute", "unsqueeze", "squeeze", "detach", "nested"])
    @pytest.mark.parametrize("mutate_base", [False, True])
    def test_view_mutation(self, view, mutate_base, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            __constants__ = ["view", "mutate_base"]

            def __init__(self):
                super().__init__()
                self.view = view
                self.mutate_base = mutate_base

            def forward(self, data):
                base = data.clone()
                if self.view == "view":
                    alias = base.view(3, 20)
                elif self.view == "transpose":
                    alias = base.transpose(0, 2)
                elif self.view == "permute":
                    alias = base.permute(2, 0, 1)
                elif self.view == "unsqueeze":
                    alias = base.unsqueeze(0)
                elif self.view == "squeeze":
                    alias = base.unsqueeze(0).squeeze(0)
                elif self.view == "detach":
                    alias = base.detach()
                else:
                    alias = base.transpose(0, 2)[1:4, :, 1:].unsqueeze(0)
                before = alias + 1
                if self.mutate_base:
                    base.add_(2)
                else:
                    alias.add_(2)
                return base, alias, before, alias * 3

        self._test(Model(), "aten::add_", ie_device, precision, ir_version, freeze_model=False)

    def test_sibling_views(self, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, data):
                base = data.clone()
                left = base[:, :3]
                right = base[:, 1:]
                before = right + 1
                left.add_(2)
                right.mul_(3)
                left[:, 1].fill_(7)
                return base, left, right, before, right + 1

        self._test(Model(), ["aten::add_", "aten::mul_", "aten::fill_"],
                   ie_device, precision, ir_version, freeze_model=False)

    @pytest.mark.parametrize("unpack", [False, True])
    @pytest.mark.parametrize("mutate_base", [False, True])
    def test_split_view_mutation(self, unpack, mutate_base, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            __constants__ = ["unpack", "mutate_base"]

            def __init__(self):
                super().__init__()
                self.unpack = unpack
                self.mutate_base = mutate_base

            def forward(self, data):
                base = data.clone()
                parts = torch.split_with_sizes(base, [1, 3], dim=1)
                if self.unpack:
                    left, right = parts
                else:
                    left, right = parts[0], parts[1]
                before = right + 1
                if self.mutate_base:
                    base.add_(2)
                else:
                    right.add_(2)
                left.mul_(3)
                return base, left, right, before

        self._test(Model(), ["aten::split_with_sizes", "aten::mul_"],
                   ie_device, precision, ir_version, freeze_model=False)

    def test_list_append_then_base_mutation(self, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, data):
                base = data.clone()
                parts = [base, base[:, :2]]
                parts.append(base[:, 2:])
                before = parts[2] + 1
                base.add_(2)
                return parts[0], parts[1], parts[2], before

        self._test(Model(), ["aten::append", "aten::add_"],
                   ie_device, precision, ir_version, freeze_model=False, fx_kind="aten.add_")

    @pytest.mark.parametrize("mutate_base", [False, True])
    def test_loop_view_mutation(self, mutate_base, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            __constants__ = ["mutate_base"]

            def __init__(self):
                super().__init__()
                self.mutate_base = mutate_base

            def forward(self, data):
                base = data.clone()
                view = base.transpose(0, 2)
                before = view + 1
                for _ in range(2):
                    if self.mutate_base:
                        base.add_(2)
                    else:
                        view.add_(2)
                return base, view, before

        self._test(Model(), ["prim::Loop", "aten::add_"],
                   ie_device, precision, ir_version, freeze_model=False, fx_kind="aten.add_")

    @pytest.mark.parametrize("change_base", [False, True])
    @pytest.mark.parametrize("metadata_op", ["unsqueeze", "squeeze", "transpose"])
    def test_metadata_mutation(self, change_base, metadata_op, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            __constants__ = ["change_base", "metadata_op"]

            def __init__(self):
                super().__init__()
                self.change_base = change_base
                self.metadata_op = metadata_op

            def forward(self, data):
                base = data.clone().unsqueeze(0)
                view = base[:, :, 1:]
                before = view + 1
                target = base if self.change_base else view
                if self.metadata_op == "unsqueeze":
                    target.unsqueeze_(0)
                elif self.metadata_op == "squeeze":
                    target.squeeze_(0)
                else:
                    target.transpose_(1, 3)
                target.add_(2)
                return base, view, before

        self._test(Model(), [f"aten::{metadata_op}_", "aten::add_"],
                   ie_device, precision, ir_version, freeze_model=False)

    @pytest.mark.parametrize("view", ["real", "imag", "view_as_real", "view_as_complex"])
    @pytest.mark.parametrize("mutate_base", [False, True])
    def test_complex_view_mutation(self, view, mutate_base, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            __constants__ = ["view", "mutate_base"]

            def __init__(self):
                super().__init__()
                self.view = view
                self.mutate_base = mutate_base

            def forward(self, data):
                if self.view == "view_as_complex":
                    base = torch.stack((data, data * 2), dim=-1)
                    alias = torch.view_as_complex(base)
                else:
                    base = torch.complex(data, data * 2)
                    if self.view == "real":
                        alias = base.real
                    elif self.view == "imag":
                        alias = base.imag
                    else:
                        alias = torch.view_as_real(base)
                before = alias + 1
                if self.mutate_base:
                    base.add_(2)
                else:
                    alias.add_(2)
                if self.view == "view_as_complex":
                    return base, torch.view_as_real(alias), torch.view_as_real(before)
                return torch.view_as_real(base), alias, before

        self._test(Model(), "aten::add_", ie_device, precision, ir_version, freeze_model=False)
