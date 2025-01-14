# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestNonZero(PytorchLayerTest):
    def _prepare_input(self, mask_fill='ones', mask_dtype=bool):
        input_shape = [2, 10, 2]
        mask = np.zeros(input_shape).astype(mask_dtype)
        if mask_fill == 'ones':
            mask = np.ones(input_shape).astype(mask_dtype)
        if mask_fill == 'random':
            idx = np.random.choice(10, 5)
            mask[:, idx, 1] = 1
        return (mask,)

    def create_model(self, as_tuple):
        import torch

        class aten_nonzero(torch.nn.Module):

            def forward(self, cond):
                return torch.nonzero(cond)

        class aten_nonzero_numpy(torch.nn.Module):

            def forward(self, cond):
                return torch.nonzero(cond, as_tuple=True)

        ref_net = None

        if not as_tuple:
            return aten_nonzero(), ref_net, "aten::nonzero"
        return aten_nonzero_numpy(), ref_net, "aten::nonzero_numpy"

    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])  # np.float32 incorrectly casted to bool
    @pytest.mark.parametrize("mask_dtype", [np.uint8, bool])
    @pytest.mark.parametrize("as_tuple", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_nonzero(self, mask_fill, mask_dtype, as_tuple, ie_device, precision, ir_version):
        self._test(*self.create_model(as_tuple),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'mask_fill': mask_fill, 'mask_dtype': mask_dtype}, trace_model=as_tuple)
