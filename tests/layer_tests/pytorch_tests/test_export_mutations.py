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
    """Mutation cases which only exist in the exported ATen graph.

    View mutations shared with TorchScript are covered by test_aliases.py::TestViewMutations,
    which the harness also runs in export mode.
    """

    @pytest.fixture(autouse=True)
    def forbid_decompositions(self):
        # Conversion must handle the exported ATen ops directly.
        with patch.object(torch.export.ExportedProgram, "run_decompositions",
                          side_effect=AssertionError("Conversion must not run decompositions")):
            yield

    def _prepare_input(self):
        return (self.random.randn(3, 4, 5).astype("float32"),)

    @pytest.mark.parametrize("view", ["slice", "select", "copy"])
    @pytest.mark.parametrize("mutate_base", [False, True])
    def test_view_mutation(self, view, mutate_base, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                base = x.clone()
                if view == "slice":
                    alias = base[:, 1:3, ::2]
                elif view == "select":
                    alias = base[1]
                else:
                    # A non-contiguous reshape allocates new storage.
                    alias = base.transpose(0, 2).reshape(-1)
                before = alias + 1
                if mutate_base:
                    base.add_(2)
                else:
                    alias.add_(2)
                return base, alias, before, alias * 3

        self._test(Model(), "aten::add_", ie_device, precision, ir_version)

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
