# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestReshapeAs(PytorchLayerTest):

    def _prepare_input(self, shape1, shape2):
        return (np.ones(shape1, dtype=np.float32), np.ones(shape2, dtype=np.float32))

    def create_model(self, op):
        class aten_reshape_as(torch.nn.Module):
            def __init__(self, op) -> None:
                super().__init__()
                if op == "view_as":
                    self.forward = self.forward_view

            def forward(self, input_tensor, shape_tensor):
                return input_tensor.reshape_as(shape_tensor)

            def forward_view(self, input_tensor, shape_tensor):
                return input_tensor.view_as(shape_tensor)

        ref_net = None

        return aten_reshape_as(op), ref_net, f"aten::{op}"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("op", ["reshape_as", "view_as"])
    @pytest.mark.parametrize('input_tensor_shapes',( ((3, 6), (2, 9)), ((2, 2, 3), (6, 2)), ((6, 2), (2, 2, 3))))
    def test_reshape_as(self, op, input_tensor_shapes, ie_device, precision, ir_version):
        self._test(*self.create_model(op), ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"shape1": input_tensor_shapes[0], "shape2": input_tensor_shapes[1]})
