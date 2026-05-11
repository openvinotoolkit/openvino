# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from packaging import version

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export

# torch.where with a uint8 condition tensor is deprecated in PyTorch 2.9.
_WHERE_UINT8_DEPRECATED = version.parse(torch.__version__) >= version.parse("2.9.0")


class Testwhere(PytorchLayerTest):
    def _prepare_input(self, mask_fill='ones', mask_dtype=bool, return_x_y=False, x_dtype="float32", y_dtype=None, scalar_cond=False):
        input_shape = [2, 10]
        if scalar_cond:
            mask = np.array(0 if mask_fill == 'zeros' else 1).astype(mask_dtype)
        else:
            mask = np.zeros(input_shape).astype(mask_dtype)
            if mask_fill == 'ones':
                mask = np.ones(input_shape).astype(mask_dtype)
            if mask_fill == 'random':
                idx = self.random.choice(10, 5)
                mask[:, idx] = 1
        x = self.random.randn(*input_shape, dtype=x_dtype)
        y = self.random.randn(*input_shape, dtype=y_dtype or x_dtype)
        return (mask,) if not return_x_y else (mask, x, y)

    def create_model(self, as_non_zero, dtypes=None):
        import torch

        dtype_map = {
            "float32": torch.float32,
            "int32": torch.int32
        }

        torch_dtypes = None
        if dtypes:
            torch_dtypes = (dtype_map[dtypes[0]], dtype_map[dtypes[1]])

        class aten_where(torch.nn.Module):
            def __init__(self, dtypes) -> None:
                super().__init__()
                self.x_dtype = dtypes[0]
                self.y_dtype = dtypes[1]


            def forward(self, cond, x, y):
                return torch.where(cond, x.to(self.x_dtype), y.to(self.y_dtype))

        class aten_where_as_nonzero(torch.nn.Module):
            def forward(self, cond):
                return torch.where(cond)

        class aten_where_as_nonzero_getitem(torch.nn.Module):
            def forward(self, cond: torch.Tensor):
                return torch.where(cond)[0]

        if as_non_zero == 'scripted':
            return aten_where_as_nonzero_getitem(), "aten::where"
        if as_non_zero:
            return aten_where_as_nonzero(), "aten::where"
        return aten_where(torch_dtypes), "aten::where"

    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("mask_dtype", [skip_if_export(np.uint8, reason="torch.export requires bool predicate for where"), bool])
    @pytest.mark.parametrize("x_dtype", ["float32", "int32"])
    @pytest.mark.parametrize("y_dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_where(self, mask_fill, mask_dtype, x_dtype, y_dtype, ie_device, precision, ir_version):
        if mask_dtype == np.uint8 and _WHERE_UINT8_DEPRECATED:
            pytest.skip("torch.where with uint8 condition is deprecated in PyTorch 2.9; use bool condition instead")
        self._test(*self.create_model(False, dtypes=(x_dtype, y_dtype)),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'mask_fill': mask_fill,
                       'mask_dtype': mask_dtype,
                       'return_x_y': True,
                       "x_dtype": x_dtype,
                       "y_dtype": y_dtype
                       })

    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("mask_dtype", [np.uint8, bool])  # np.float32 incorrectly casted to bool
    @pytest.mark.parametrize("x_dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_where_as_nonzero(self, mask_fill, mask_dtype, x_dtype, ie_device, precision, ir_version):
        if mask_dtype == np.uint8 and _WHERE_UINT8_DEPRECATED:
            pytest.skip("torch.where with uint8 condition is deprecated in PyTorch 2.9; use bool condition instead")
        # torch.jit.trace is used here (trace_model=True) so that example inputs annotate concrete
        # shapes, giving translate_where a static input rank to split NonZero output into per-dim
        # index tensors.  Without example inputs torch.jit.script leaves rank dynamic and
        # translate_where cannot determine the number of output tensors at conversion time.
        self._test(*self.create_model(True),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'mask_fill': mask_fill,
                       'mask_dtype': mask_dtype,
                       'return_x_y': False,
                       "x_dtype": x_dtype,
                       },
                   trace_model=True)

    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("cond_dtype", [np.float32, np.int32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_where_as_nonzero_nonbool_cond(self, mask_fill, cond_dtype, ie_device, precision, ir_version):
        # aten::where(cond) must accept non-boolean condition tensors (float32, int32).
        # NonZero treats zero as False and any nonzero value as True, matching PyTorch semantics.
        self._test(*self.create_model(True),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'mask_fill': mask_fill,
                       'mask_dtype': cond_dtype,
                       'return_x_y': False,
                       'x_dtype': "float32",
                       },
                   trace_model=True)

    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("mask_dtype", [bool])
    @pytest.mark.parametrize("x_dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_where_as_nonzero_export(self, mask_fill, mask_dtype, x_dtype, ie_device, precision, ir_version):
        # torch.export produces aten.where.default which previously had no registered translator,
        # causing OpConversionFailure.
        self._test(*self.create_model(True),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'mask_fill': mask_fill,
                       'mask_dtype': mask_dtype,
                       'return_x_y': False,
                       "x_dtype": x_dtype,
                       })

    @pytest.mark.parametrize("cond_val", ['zeros', 'ones'])
    @pytest.mark.parametrize("x_dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_where_scalar_cond(self, cond_val, x_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(False, dtypes=(x_dtype, x_dtype)),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'mask_fill': cond_val,
                       'mask_dtype': bool,
                       'return_x_y': True,
                       'x_dtype': x_dtype,
                       'scalar_cond': True,
                   })

    @pytest.mark.parametrize("mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("x_dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_where_as_nonzero_scripted(self, mask_fill, x_dtype, ie_device, precision, ir_version):
        # Tests translate_where's input_is_none(1) branch via torch.jit.script.
        # The model returns torch.where(cond)[0], producing aten::where → aten::__getitem__
        self._test(*self.create_model('scripted'),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'mask_fill': mask_fill,
                       'mask_dtype': bool,
                       'return_x_y': False,
                       'x_dtype': x_dtype,
                   },
                   trace_model=False,
                   dynamic_shapes=False,
                   use_convert_model=True,
                   freeze_model=False)

