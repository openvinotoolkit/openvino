# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestBroadcastTensors(PytorchLayerTest):
    def _prepare_input(self, x_shape, y_shape, z_shape, x_dtype, y_dtype, z_dtype):
        import numpy as np
        return (
            np.random.randn(*x_shape).astype(x_dtype), 
            np.random.randn(*y_shape).astype(y_dtype), 
            np.random.randn(*z_shape).astype(z_dtype))

    def create_model(self):
        import torch

        class aten_broadcast_tensors(torch.nn.Module):
            def __init__(self):
                super(aten_broadcast_tensors, self).__init__()

            def forward(self, x, y, z):
                x1, y1, z1 = torch.broadcast_tensors(x, y, z)
                return x1, y1, z1

        ref_net = None

        return aten_broadcast_tensors(), ref_net, ("prim::ListConstruct", "aten::broadcast_tensors", "prim::ListUnpack")

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("x_shape", [[1, ], [2, 1], [2, 2, 1]])
    @pytest.mark.parametrize("y_shape", [[2, ], [1, 2], [1, 2, 1]])
    @pytest.mark.parametrize("z_shape", [[1, 2], [2, 2], [1, 2, 1, 1]])
    @pytest.mark.parametrize("x_dtype", ["float32", "int32"])
    @pytest.mark.parametrize("y_dtype", ["float32", "int32"])
    @pytest.mark.parametrize("z_dtype", ["float32", "int32"])
    def test_broadcast_tensors(self, x_shape, y_shape, z_shape, x_dtype, y_dtype, z_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, kwargs_to_prepare_input={
            "x_shape": x_shape, "x_dtype": x_dtype,
            "y_shape": y_shape, "y_dtype": y_dtype,
            "z_shape": z_shape, "z_dtype": z_dtype,
        })
