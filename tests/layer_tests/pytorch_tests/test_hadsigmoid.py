# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestHardSigmoid(PytorchLayerTest):
    def _prepare_input(self, shape, dtype):
        import numpy as np
        return (np.random.randn(*shape).astype(dtype),)

    def create_model(self, inplace):
        import torch
        import torch.nn.functional as F

        class aten_hardsigmoid(torch.nn.Module):
            def __init__(self, inplace):
                super(aten_hardsigmoid, self).__init__()
                self.inplace = inplace

            def forward(self, x):
                return F.hardsigmoid(x, self.inplace), x

        ref_net = None

        return aten_hardsigmoid(inplace), ref_net, "aten::hardsigmoid" if not inplace else "aten::hardsigmoid_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("shape", [[1, 10], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    def test_hardsigmoid(self, shape, dtype, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(inplace), ie_device, precision, ir_version, kwargs_to_prepare_input={"shape": shape, "dtype": dtype})
