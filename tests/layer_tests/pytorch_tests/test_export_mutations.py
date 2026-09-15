# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.skipif(not PytorchLayerTest.use_torch_export(), reason="Tests the exported ATen graph")
@pytest.mark.precommit_torch_export
@pytest.mark.nightly
class TestExportMutations(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(3, 4, 5).astype("float32"),)

    @pytest.mark.parametrize("view", ["slice", "select", "view", "transpose", "permute", "squeeze", "detach", "copy"])
    @pytest.mark.parametrize("mutate_base", [False, True])
    def test_view_mutation(self, view, mutate_base, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                base = x.clone()
                if view == "slice":
                    alias = base[:, 1:3, ::2]
                elif view == "select":
                    alias = base[1]
                elif view == "view":
                    alias = base.view(3, 20)
                elif view == "transpose":
                    alias = base.transpose(0, 2)
                elif view == "permute":
                    alias = base.permute(2, 0, 1)
                elif view == "squeeze":
                    alias = base.unsqueeze(0).squeeze(0)
                elif view == "detach":
                    alias = base.detach()
                else:
                    # A non-contiguous reshape allocates new storage.
                    alias = base.transpose(0, 2).reshape(-1)
                before = alias + 1
                if mutate_base:
                    base.add_(2)
                else:
                    alias.add_(2)
                return base, alias, before, alias * 3

        with patch.object(torch.export.ExportedProgram, "run_decompositions",
                          side_effect=AssertionError("Conversion must not run decompositions")):
            self._test(Model(), "aten::add_", ie_device, precision, ir_version)

    def test_sibling_views(self, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                base = x.clone()
                left = base[:, :3]
                right = base[:, 1:]
                before = right + 1
                left.add_(2)
                right.mul_(3)
                left[:, 1].fill_(7)
                return base, left, right, before, right + 1

        self._test(Model(), ["aten::add_", "aten::mul_", "aten::fill_"], ie_device, precision, ir_version)

    def test_inplace_result_alias(self, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                result = x.add_(2)
                before = result * 2
                x.mul_(3)
                result.sub_(1)
                return x, result, before

        self._test(Model(), ["aten::add_", "aten::mul_", "aten::sub_"], ie_device, precision, ir_version)

    @pytest.mark.parametrize("view", ["split", "chunk", "unbind", "indices", "sections", "sizes"])
    @pytest.mark.parametrize("mutate_base", [False, True])
    def test_list_view_mutation(self, view, mutate_base, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                base = x.clone()
                if view == "split":
                    parts = base.split(1, dim=1)
                elif view == "chunk":
                    parts = base.chunk(2, dim=1)
                elif view == "unbind":
                    parts = base.unbind(dim=1)
                elif view == "indices":
                    parts = base.tensor_split([1, 3], dim=1)
                elif view == "sections":
                    parts = base.tensor_split(3, dim=1)
                else:
                    parts = base.split([1, 3], dim=1)
                before = parts[1] + 1
                if mutate_base:
                    base.add_(2)
                else:
                    parts[1].add_(2)
                parts[0].mul_(3)
                return base, parts, before

        self._test(Model(), ["aten::add_", "aten::mul_"], ie_device, precision, ir_version,
                   dynamic_shapes_for_export={"x": {0: torch.export.Dim("batch", min=2)}})

    def test_out_view_mutation(self, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                base = torch.zeros_like(x)
                view = base[:, 1:3]
                before = base + 3
                result = torch.add(x[:, 1:3], 1, out=view)
                base.add_(2)
                result.mul_(2)
                return base, view, result, before

        self._test(Model(), ["aten::add", "aten::add_", "aten::mul_"], ie_device, precision, ir_version,
                   fx_kind=["aten.add.out", "aten.add_.Tensor", "aten.mul_.Tensor"])

    @pytest.mark.parametrize("change_base", [False, True])
    @pytest.mark.parametrize("metadata_op", ["unsqueeze", "squeeze", "transpose"])
    def test_metadata_then_data_mutation(self, change_base, metadata_op, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                base = x.clone().unsqueeze(0)
                view = base[:, :, 1:]
                before = view + 1
                target = base if change_base else view
                if metadata_op == "unsqueeze":
                    target.unsqueeze_(0)
                elif metadata_op == "squeeze":
                    target.squeeze_(0)
                else:
                    target.transpose_(1, 3)
                target.add_(2)
                return base, view, before

        self._test(Model(), [f"aten::{metadata_op}_", "aten::add_"], ie_device, precision, ir_version)

    @pytest.mark.parametrize("view", ["real", "imag", "view_as_real", "view_as_complex"])
    @pytest.mark.parametrize("mutate_base", [False, True])
    def test_complex_view_mutation(self, view, mutate_base, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                if view == "view_as_complex":
                    base = torch.stack((x, x * 2), dim=-1)
                    alias = torch.view_as_complex(base)
                else:
                    base = torch.complex(x, x * 2)
                    if view == "real":
                        alias = base.real
                    elif view == "imag":
                        alias = base.imag
                    else:
                        alias = torch.view_as_real(base)
                before = alias + 1
                if mutate_base:
                    base.add_(2)
                else:
                    alias.add_(2)
                return tuple(torch.view_as_real(value) if value.is_complex() else value
                             for value in (base, alias, before))

        self._test(Model(), "aten::add_", ie_device, precision, ir_version)
