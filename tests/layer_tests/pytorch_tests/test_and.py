# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestAnd(PytorchLayerTest):

    def _prepare_input(self):
        return self.input_data

    def create_model_tensor_input(self):
        class aten_and_tensor(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, tensor_a, tensor_b):
                return tensor_a & tensor_b

        ref_net = None

        return aten_and_tensor(), ref_net, "aten::__and__"
    
    def create_model_bool_input(self):
        class aten_and_bool(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, bool_a: bool, bool_b: bool):
                return bool_a & bool_b

        ref_net = None

        return aten_and_bool(), ref_net, "aten::__and__"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_and_tensor(self, ie_device, precision, ir_version):
        self.input_data = (np.array([True, False, False], dtype=np.bool_), np.array(
            [True, True, False], dtype=np.bool_))
        self._test(*self.create_model_tensor_input(),
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_and_bool(self, ie_device, precision, ir_version):
        self.input_data = (np.array(True, dtype=np.bool_),
                           np.array(True, dtype=np.bool_))
        self._test(*self.create_model_bool_input(),
                   ie_device, precision, ir_version)
