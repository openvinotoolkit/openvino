# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestFormat(PytorchLayerTest):
    def _prepare_input(self, input_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self):
        class aten_format_tensor_return(torch.nn.Module):
            def forward(self, x):
                s = "{}".format(1)
                t = x + 1
                return (t, s)[0]

        ref_net = None
        op_in_graph = ["aten::format"]
        return aten_format_tensor_return(), ref_net, op_in_graph

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [
        ([2, 3],),
        ([3, 4, 5],),
    ])
    def test_format(self, input_shape, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"input_shape": input_shape})
