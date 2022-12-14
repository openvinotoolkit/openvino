import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np


class Testwhere(PytorchLayerTest):
    def _prepare_input(self, mask_fill='ones', mask_dtype=bool, return_x_y=False):
        input_shape = [2, 10]
        mask = np.zeros(input_shape).astype(mask_dtype)
        if mask_fill == 'ones':
            mask = np.ones(input_shape).astype(mask_dtype)
        if mask_fill == 'random':
            idx  = np.random.choice(10, 5)
            mask[:, idx] = 1
        x = np.random.randn(*input_shape)
        y = np.random.randn(*input_shape)
        return (mask, ) if not return_x_y else (mask, x, y)

    def create_model(self, as_non_zero):
        import torch

        class aten_where(torch.nn.Module):
            def forward(self, cond, x, y):
                return torch.where(cond, x, y)

        class aten_where_as_nonzero(torch.nn.Module):
            def forward(self, cond):
                return torch.where(cond)

        ref_net = None

        if as_non_zero:
            return aten_where_as_nonzero(), ref_net, "aten::where"
        return aten_where(), ref_net, "aten::where"

    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("mask_dtype", [np.uint8, bool]) # np.float32 incorrectly casted to bool
    @pytest.mark.nightly
    def test_where(self, mask_fill, mask_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(False),
                   ie_device, precision, ir_version, kwargs_to_prepare_input={'mask_fill': mask_fill, 'mask_dtype': mask_dtype, 'return_x_y': True})

    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("mask_dtype", [np.uint8, bool]) # np.float32 incorrectly casted to bool
    @pytest.mark.nightly
    def test_where_as_nonzero(self, mask_fill, mask_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(True),
                   ie_device, precision, ir_version, kwargs_to_prepare_input={'mask_fill': mask_fill, 'mask_dtype': mask_dtype, 'return_x_y': False}, trace_model=True)
