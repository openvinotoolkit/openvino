# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestBool(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randint(0, 10, 1).astype(np.int32),)

    def create_model(self, input_type):
        import torch

        class prim_bool(torch.nn.Module):
            def __init__(self, input_type):
                super(prim_bool, self).__init__()
                self.forward = self.forward_tensor if input_type != "scalar" else self.forward_scalar

            def forward_tensor(self, x):
                return bool(x)

            def forward_scalar(self, x:int):
                return bool(x)

        ref_net = None

        return prim_bool(input_type), ref_net, "aten::Bool"

    @pytest.mark.parametrize("input_type", ["tensor", "scalar"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bool(self, ie_device, precision, ir_version, input_type):
        self._test(*self.create_model(input_type), ie_device, precision, ir_version)
