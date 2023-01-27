# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSize(PytorchLayerTest):
    def _prepare_input(self, input_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self, case):
        import torch

        class aten_size(torch.nn.Module):
            def forward(self, x):
                return x.shape

        class aten_size_get_item(torch.nn.Module):
            def forward(self, x):
                return x.shape[0]

        ref_net = None

        op_cls = {
            "size": (aten_size, "aten::size"),
            "size_with_getitem": (aten_size_get_item, ["aten::size", "aten::__getitem__"])
        }
        op, op_in_graph = op_cls[case]

        return op(), ref_net, op_in_graph

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[1,], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
    @pytest.mark.parametrize("case", ["size", "size_with_getitem"])
    def test_silu(self, input_shape, case, ie_device, precision, ir_version):
        self._test(*self.create_model(case), ie_device, precision, ir_version, kwargs_to_prepare_input={"input_shape": input_shape})