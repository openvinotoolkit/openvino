# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestEq(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_array.astype(self.input_type), self.other_array.astype(self.other_type))

    def create_model(self):
        import torch

        class aten_eq(torch.nn.Module):
            def __init__(self):
                super(aten_eq, self).__init__()

            def forward(self, input_tensor, other_tensor):
                return torch.eq(input_tensor, other_tensor)

        ref_net = None

        return aten_eq(), ref_net, "aten::eq"

    @pytest.mark.parametrize(("input_array", "other_array"), [
        [np.array([[1, 2], [3, 4]]), np.array([[1, 1], [4, 4]])],
        [np.array([1, 2]), np.array([1, 2])],
        [np.array([[[6, 1], [3, 4]]]), np.array([[1, 1], [4, 4]])],
        [np.array([7, 4.1, 2.1, 8.9]), np.array([0.5, 4.1, 2.1, 15.3])],
        [np.array([-15, -31.1, -18.2]), np.array([14, -31.1, -18.2])],
        # check broadcast
        [np.ones(shape=(5, 3, 4, 1)), np.ones(shape=(3, 4, 1))]
    ])
    @pytest.mark.parametrize(("types"), [
        (np.float32, np.float32),
        (np.int32, np.int32),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_eq_pt_spec(self, input_array, other_array, types, ie_device, precision, ir_version):
        self.input_array = input_array 
        self.input_type = types[0]
        self.other_array = other_array
        self.other_type = types[1]
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)
