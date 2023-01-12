# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np
import torch


@pytest.mark.parametrize('input_tesnors', ((np.ones((3, 6)), np.ones((2, 9))),
                                           (np.ones((2, 2, 3)), np.ones((6, 2))),
                                           (np.ones((6, 2)), np.ones((2, 2, 3)))))
class TestReshapeAs(PytorchLayerTest):

    def _prepare_input(self):
        return self.input_tesnors

    def create_model(self):

        class aten_reshape_as(torch.nn.Module):

            def forward(self, input_tensor, shape_tensor):
                return input_tensor.reshape_as(shape_tensor)

        ref_net = None

        return aten_reshape_as(), ref_net, "aten::reshape_as"

    @pytest.mark.nightly
    def test_reshape_as(self, ie_device, precision, ir_version, input_tesnors):
        self.input_tesnors = input_tesnors
        self._test(*self.create_model(), ie_device, precision, ir_version)
