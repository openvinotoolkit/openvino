# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestLinalgCross(PytorchLayerTest):
    def _prepare_input(self, x_shape, y_shape, out, dtype):
        import numpy as np
        x = np.random.randn(*x_shape).astype(dtype)
        y = np.random.randn(*y_shape).astype(dtype)
        if not out:
            return (x, y)
        return (x, y, np.zeros(np.maximum(np.array(x_shape), np.array(y_shape)).tolist(), dtype=dtype))

    def create_model(self, dim, out):
        import torch

        class aten_linalg_cross(torch.nn.Module):
            def __init__(self, dim, out):
                super(aten_linalg_cross, self).__init__()
                if dim is None:
                    self.forward = self.forward_no_dim_no_out if not out else self.forward_no_dim_out
                elif out:
                    self.forward = self.forward_out
                self.dim = dim

            def forward(self, x, y):
                return torch.linalg.cross(x, y, dim=self.dim)

            def forward_out(self, x, y, out):
                return torch.linalg.cross(x, y, dim=self.dim, out=out), out

            def forward_no_dim_out(self, x, y, out):
                return torch.linalg.cross(x, y, out=out), out

            def forward_no_dim_no_out(self, x, y):
                return torch.linalg.cross(x, y)

        ref_net = None

        return aten_linalg_cross(dim, out), ref_net, "aten::linalg_cross"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("x_shape,y_shape,dim", [
        ((4, 3), (4, 3), None),
        ((1, 3), (4, 3), -1),
        ((4, 3), (1, 3), 1),
        ((3, 5), (3, 5), 0),
        ((2, 3, 4), (2, 3, 4), 1)
    ])
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    @pytest.mark.parametrize("out", [True, False])
    def test_linalg_cross(self, x_shape, y_shape, dim, out, dtype, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dim, out), ie_device, precision, ir_version, use_convert_model=True,
            kwargs_to_prepare_input={"x_shape": x_shape,
                                     "y_shape": y_shape,
                                     "out": out,
                                     'dtype': dtype},
            dynamic_shapes=ie_device != "GPU")


class TestCross(PytorchLayerTest):
    def _prepare_input(self, x_shape, y_shape, out, dtype):
        import numpy as np
        x = np.random.randn(*x_shape).astype(dtype)
        y = np.random.randn(*y_shape).astype(dtype)
        if not out:
            return (x, y)
        return (x, y, np.zeros(np.maximum(np.array(x_shape), np.array(y_shape)).tolist(), dtype=dtype))

    def create_model(self, dim, out, shape):
        import torch

        class aten_cross(torch.nn.Module):
            def __init__(self, dim, out, shape):
                super(aten_cross, self).__init__()
                if dim is None:
                    self.forward = self.forward_no_dim_no_out if not out else self.forward_no_dim_out
                elif out:
                    self.forward = self.forward_out
                self.dim = dim
                self.shape = shape

            def forward(self, x, y):
                return torch.cross(x, y, dim=self.dim)

            def forward_out(self, x, y, out):
                return torch.cross(x, y, dim=self.dim, out=out), out

            def forward_no_dim_out(self, x, y, out):
                x = torch.reshape(x, self.shape)
                return torch.cross(x, y, out=out), out

            def forward_no_dim_no_out(self, x, y):
                x = torch.reshape(x, self.shape)
                return torch.cross(x, y)

        ref_net = None

        return aten_cross(dim, out, shape), ref_net, "aten::cross"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("x_shape,y_shape,dim", [
        ((1, 3), (4, 3), -1),
        ((4, 3), (1, 3), 1),
        ((3, 5), (3, 5), 0),
        ((2, 3, 4), (2, 3, 4), 1),
        ((3, 1), (3, 4), None),
        ((4, 3), (4, 3), None),
        ((2, 3, 4), (2, 3, 4), None),
    ])
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    def test_linalg_cross(self, x_shape, y_shape, dim, out, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dim, out, x_shape), ie_device, precision, ir_version,
                   use_convert_model=True,
                   kwargs_to_prepare_input={"x_shape": x_shape,
                                            "y_shape": y_shape,
                                            "out": out,
                                            "dtype": dtype},
                   dynamic_shapes=ie_device != "GPU")
