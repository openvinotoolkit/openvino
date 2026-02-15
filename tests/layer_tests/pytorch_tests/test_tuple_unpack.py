# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestTupleUnpack(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 4, 6, 8).astype(np.float32),)

    def create_model(self):

        import torch

        class TupleArgument(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dims = (1, 2, 3, 4)

            def forward(self, input_data):
                N, _, H, W = self.dims
                return input_data * N * H * W

        return TupleArgument(), None, "prim::TupleUnpack"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_touple_unpack(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=False, freeze_model=False)
