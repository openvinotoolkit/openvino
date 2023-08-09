# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestTranspose(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 3, 4, 5).astype(np.float32),)

    def create_model(self, dim0, dim1):
        import torch

        class aten_transpose(torch.nn.Module):
            def __init__(self, dim0, dim1):
                super(aten_transpose, self).__init__()
                self.dim0 = dim0
                self.dim1 = dim1

            def forward(self, x):
                return torch.transpose(x, self.dim0, self.dim1)

        ref_net = None

        return aten_transpose(dim0, dim1), ref_net, "aten::transpose"

    @pytest.mark.parametrize("dim0", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.parametrize("dim1", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_transpose(self, dim0, dim1, ie_device, precision, ir_version):
        self._test(*self.create_model(dim0, dim1),
                   ie_device, precision, ir_version)


class TestTSmall(PytorchLayerTest):
    def _prepare_input(self, num_dims=2, input_dtype="float32"):
        import numpy as np
        shape = (2, 3)
        if num_dims == 0:
            return (np.array(num_dims).astype(input_dtype), )
        return (np.random.randn(*shape[:num_dims]).astype(input_dtype),)

    def create_model(self, num_dims=2, inplace=False):
        import torch

        class aten_transpose(torch.nn.Module):
            def __init__(self, inplace):
                super(aten_transpose, self).__init__()
                if inplace:
                    self.forward = self.forward_inplace

            def forward(self, x):
                return x.t(), x

            def forward_inplace(self, x):
                return x.t_(), x

        ref_net = None

        return aten_transpose(inplace), ref_net, "aten::t" if not inplace else "aten::t_" 

    @pytest.mark.parametrize("num_dims", [0, 1, 2])
    @pytest.mark.parametrize("input_dtype", ["float32", "int32"])
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_t_small(self, num_dims, input_dtype, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(num_dims, inplace),
                   ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"num_dims": num_dims, "input_dtype": input_dtype})
