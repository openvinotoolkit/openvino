# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_data', [(np.array(True, dtype=np.bool_), np.array(True, dtype=np.bool_)),
                                        (np.array([True, False, False], dtype=np.bool_), np.array([True, True, False], dtype=np.bool_)),
                                        (np.array([0, 1, 2]), np.array([0, 0, 3]))])
class TestAnd(PytorchLayerTest):

    def _prepare_input(self):
        return self.input_data
        # return (np.array(True, dtype=np.bool_), np.array(True, dtype=np.bool_))
        # return (np.array([True, False, False], dtype=np.bool_), np.array([True, True, False], dtype=np.bool_))
        # return (np.random.randn(2,2,3), np.random.randn(2,2,3))
        # jeżeli to działa tylko na boolach, to można to rzutować wprost na operator z OV.

    def create_model(self):
        class aten_and(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, tensor_a, tensor_b):
                return tensor_a.bitwise_and(tensor_b)
                # return tensor_a.__and__(tensor_b)

        ref_net = None

        return aten_and(), ref_net, "aten::__and__"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_and(self, ie_device, precision, ir_version, input_data):
        self.input_data = input_data
        self._test(*self.create_model(), ie_device, precision, ir_version)
