# Copyright (C) 2018-2026 Intel Corporation
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
            def forward(self, x, y, z):
                x1, y1, z1 = torch.broadcast_tensors(x, y, z)
                return x1, y1, z1

        return aten_broadcast_tensors(), None, ("prim::ListConstruct", "aten::broadcast_tensors", "prim::ListUnpack")

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize(("x_shape", "y_shape", "z_shape"), [
        ([1, ], [2, ], [1, 2]),        # 1D broadcasting
        ([2, 1], [1, 2], [2, 2]),      # 2D broadcasting
        ([2, 2, 1], [1, 2, 1], [1, 2, 1, 1]),  # mixed dims broadcasting
    ])
    @pytest.mark.parametrize(("x_dtype", "y_dtype", "z_dtype"), [
        ("float32", "float32", "float32"),  # homogeneous float
        ("int32", "float32", "int32"),      # mixed types
    ])
    def test_broadcast_tensors(self, x_shape, y_shape, z_shape, x_dtype, y_dtype, z_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, kwargs_to_prepare_input={
            "x_shape": x_shape, "x_dtype": x_dtype,
            "y_shape": y_shape, "y_dtype": y_dtype,
            "z_shape": z_shape, "z_dtype": z_dtype,
        },
        fx_kind="aten.broadcast_tensors.default")
