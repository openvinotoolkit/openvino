# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch
import openvino as ov

from pytorch_layer_test_class import PytorchLayerTest
import numpy as np

d2_params = [{'kernel_size': [3, 3], 'stride': 1, 'padding': 0},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': 1},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [0, 1]},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [1, 0]},
             {'kernel_size': [3, 3], 'stride': [2, 1], 'padding': 0},
             {'kernel_size': [2, 1], 'stride': [2, 1], 'padding': 0},
             {'kernel_size': [2, 1], 'stride': None, 'padding': 0},
             {'kernel_size': [2, 1], 'stride': [], 'padding': 0},
             {'kernel_size': [8, 8], 'stride': [8, 4], 'padding': 1},
             ]

d1_params = [{'kernel_size': 3, 'stride': 1, 'padding': 0},
             {'kernel_size': (4,), 'stride': 1, 'padding': 1},
             {'kernel_size': 4, 'stride': (5,), 'padding': 2},
             {'kernel_size': 4, 'stride': None, 'padding': 0},
             ]
d3_params = [{'kernel_size': [3, 3, 3], 'stride': 1, 'padding': 0},
             {'kernel_size': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 1},
             {'kernel_size': [3, 3, 3], 'stride': [
                 3, 3, 3], 'padding': [0, 0, 0]},
             {'kernel_size': [3, 2, 1], 'stride': [
                 3, 1, 1], 'padding': [0, 0, 0]},
             {'kernel_size': [3, 2, 1], 'stride': None, 'padding': [0, 0, 0]},
             ]


class TestPooling(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, op_type, kernel_size, stride, padding, dilation=1, ceil_mode=True, count_include_pad=True, dtype=torch.float32):
        class aten_avg_pooling_base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.ceil_mode = ceil_mode
                self.count_include_pad = count_include_pad

            def forward(self, x):
                pass

        class aten_max_pooling_base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.ceil_mode = ceil_mode
                self.dtype = dtype

            def forward(self, x):
                pass

        class aten_avg_pool2d(aten_avg_pooling_base):
            def forward(self, x):
                return torch.nn.functional.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                                      self.count_include_pad)

        class aten_avg_pool3d(aten_avg_pooling_base):
            def forward(self, x):
                return torch.nn.functional.avg_pool3d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                                      self.count_include_pad)

        class aten_avg_pool1d(aten_avg_pooling_base):
            def forward(self, x):
                return torch.nn.functional.avg_pool1d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                                      self.count_include_pad)

        class aten_max_pool2d(aten_max_pooling_base):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x.to(self.dtype), self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode)

        class aten_max_pool3d(aten_max_pooling_base):
            def forward(self, x):
                return torch.nn.functional.max_pool3d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode)

        class aten_max_pool1d(aten_max_pooling_base):
            def forward(self, x):
                return torch.nn.functional.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode)

        class aten_max_pool2d_indices(aten_max_pooling_base):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode, return_indices=True)

        class aten_max_pool3d_indices(aten_max_pooling_base):
            def forward(self, x):
                return torch.nn.functional.max_pool3d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode, return_indices=True)

        class aten_max_pool1d_indices(aten_max_pooling_base):
            def forward(self, x):
                return torch.nn.functional.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode, return_indices=True)

        ops = {
            "max_pool1d": aten_max_pool1d,
            "max_pool2d": aten_max_pool2d,
            "max_pool3d": aten_max_pool3d,
            "avg_pool1d": aten_avg_pool1d,
            "avg_pool2d": aten_avg_pool2d,
            "avg_pool3d": aten_avg_pool3d,
            "max_pool1d_with_indices": aten_max_pool1d_indices,
            "max_pool2d_with_indices": aten_max_pool2d_indices,
            "max_pool3d_with_indices": aten_max_pool3d_indices,
        }

        aten_pooling = ops[op_type]

        return aten_pooling(), f"aten::{op_type}"

    @pytest.mark.parametrize("input_shape", [[1, 3, 15], [3, 15]])
    @pytest.mark.parametrize("params", d1_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_avg_pool1d(self, input_shape, params, ceil_mode, count_include_pad, ie_device, precision, ir_version, is_dynamic_shapes):
        self.input_tensor = self.random.randn(*input_shape)
        self._test(*self.create_model("avg_pool1d", **params, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
                   ie_device, precision, ir_version, trace_model=True,
                   dynamic_shapes=is_dynamic_shapes)

    @pytest.mark.parametrize("input_shape", [[1, 3, 15, 15], [3, 15, 15]])
    @pytest.mark.parametrize("params", d2_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_avg_pool2d(self, input_shape, params, ceil_mode, count_include_pad, ie_device, precision, ir_version, is_dynamic_shapes):
        if ceil_mode and count_include_pad and np.array_equal(np.array(params["kernel_size"]), np.array([8, 8])):
            pytest.xfail("Ticket - 150292")
        self.input_tensor = self.random.randn(*input_shape)
        self._test(*self.create_model("avg_pool2d", **params, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
                   ie_device, precision, ir_version, trace_model=True, freeze_model=False, dynamic_shapes=is_dynamic_shapes)

    @pytest.mark.parametrize("input_shape", [[1, 3, 15, 15, 15], [3, 15, 15, 15]])
    @pytest.mark.parametrize("params", d3_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_avg_pool3d(self, input_shape, params, ceil_mode, count_include_pad, ie_device, precision, ir_version, is_dynamic_shapes):
        self.input_tensor = self.random.randn(*input_shape)
        self._test(*self.create_model("avg_pool3d", **params, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
                   ie_device, precision, ir_version, trace_model=True,
                   dynamic_shapes=is_dynamic_shapes)

    @pytest.mark.parametrize("input_shape", [[1, 3, 15], [3, 15]])
    @pytest.mark.parametrize("params", d1_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_max_pool1d(self, input_shape, params, ceil_mode, dilation, ie_device, precision, ir_version, is_dynamic_shapes):
        self.input_tensor = self.random.randn(*input_shape)
        self._test(*self.create_model("max_pool1d", **params, ceil_mode=ceil_mode, dilation=dilation),
                   ie_device, precision, ir_version, dynamic_shapes=is_dynamic_shapes)

    @pytest.mark.parametrize("input_shape", [[1, 3, 15, 15], [3, 15, 15]])
    @pytest.mark.parametrize("params", d2_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_max_pool2d(self, input_shape, params, ceil_mode, dilation, dtype, ie_device, precision, ir_version, is_dynamic_shapes):
        to_trace = False
        if params["stride"] == []:
            to_trace = True
        self.input_tensor = self.random.randn(*input_shape)
        self._test(*self.create_model("max_pool2d", **params, ceil_mode=ceil_mode, dilation=dilation, dtype=dtype),
                   ie_device, precision, ir_version, dynamic_shapes=is_dynamic_shapes, trace_model=to_trace)

    @pytest.mark.parametrize("input_shape", [[1, 3, 15, 15, 15], [3, 15, 15, 15]])
    @pytest.mark.parametrize("params", d3_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_max_pool3d(self, input_shape, params, ceil_mode, dilation, ie_device, precision, ir_version, is_dynamic_shapes):
        self.input_tensor = self.random.randn(*input_shape)
        self._test(*self.create_model("max_pool3d", **params, ceil_mode=ceil_mode, dilation=dilation),
                   ie_device, precision, ir_version,  dynamic_shapes=is_dynamic_shapes)

    @pytest.mark.parametrize("input_shape", [[1, 3, 15], [3, 15]])
    @pytest.mark.parametrize("params", d1_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_max_pool1d_indices(self, input_shape, params, ceil_mode, dilation, ie_device, precision, ir_version, is_dynamic_shapes):
        self.input_tensor = self.random.randn(*input_shape)
        self._test(*self.create_model("max_pool1d_with_indices", **params, ceil_mode=ceil_mode, dilation=dilation),
                   ie_device, precision, ir_version, dynamic_shapes=is_dynamic_shapes)

    @pytest.mark.parametrize("input_shape", [[1, 3, 15, 15], [3, 15, 15]])
    @pytest.mark.parametrize("params", d2_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_max_pool2d_indices(self, input_shape, params, ceil_mode, dilation,  ie_device, precision, ir_version, is_dynamic_shapes):
        to_trace = False
        if params["stride"] == []:
            to_trace = True
        self.input_tensor = self.random.randn(*input_shape)
        self._test(*self.create_model("max_pool2d_with_indices", **params, ceil_mode=ceil_mode, dilation=dilation),
                   ie_device, precision, ir_version, dynamic_shapes=is_dynamic_shapes, trace_model=to_trace)

    @pytest.mark.parametrize("input_shape", [[1, 3, 15, 15, 15], [3, 15, 15, 15]])
    @pytest.mark.parametrize("params", d3_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_max_pool3d_indices(self, input_shape, params, ceil_mode, dilation, ie_device, precision, ir_version, is_dynamic_shapes):
        self.input_tensor = self.random.randn(*input_shape)
        self._test(*self.create_model("max_pool3d_with_indices", **params, ceil_mode=ceil_mode, dilation=dilation),
                   ie_device, precision, ir_version, dynamic_shapes=is_dynamic_shapes)


class TestMaxPoolDynamicKernel(PytorchLayerTest):
    """max_pool with a kernel_size built from x.size(...).

    Such a kernel is a runtime value in the traced TorchScript graph (a prim::ListConstruct of a
    constant and a ShapeOf-derived value), which the OpenVINO MaxPool op cannot represent because
    its kernel is a constructor attribute. When the dynamic window spans the full extent of a
    spatial axis (global pooling, as in the SAM-6D pose-estimation model), the frontend decomposes
    it into a ReduceMax over that axis. These tests exercise that path and compare against the
    PyTorch reference. trace_model=True is required so the kernel stays a runtime value (a plain
    integer literal would be const-folded and hit the static MaxPool path instead).
    """

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, dims, axes):
        # axes: spatial axis indices (within the rank-(dims+2) NCHW... tensor) whose kernel element
        # is the full extent x.size(axis); all other spatial axes get a static window of 1.
        class aten_max_pool_dynamic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dims = dims
                self.axes = axes

            def _kernel(self, x):
                # first spatial axis is index 2 (after N, C); for the no-batch case the harness still
                # feeds a rank (dims+1) tensor, handled by using sizes relative to the actual rank.
                first_spatial = x.dim() - self.dims
                ks = []
                for i in range(self.dims):
                    axis = first_spatial + i
                    # full extent on requested axes (referenced as absolute axis on the rank-(dims+2)
                    # tensor), window 1 otherwise
                    if (axis - x.dim()) in self.axes or axis in self.axes:
                        ks.append(x.size(axis))
                    else:
                        ks.append(1)
                return ks

            def forward(self, x):
                ks = self._kernel(x)
                if self.dims == 1:
                    return torch.nn.functional.max_pool1d(x, kernel_size=ks)
                if self.dims == 2:
                    return torch.nn.functional.max_pool2d(x, kernel_size=ks)
                return torch.nn.functional.max_pool3d(x, kernel_size=ks)

        op_name = {1: "aten::max_pool1d", 2: "aten::max_pool2d", 3: "aten::max_pool3d"}[dims]
        return aten_max_pool_dynamic(), op_name

    @pytest.mark.parametrize("input_shape,dims,axes", [
        ([1, 128, 40, 64], 2, [-1]),         # pool full last axis (SAM-6D PositionalEncoding)
        ([2, 8, 5, 7], 2, [-1]),             # smaller, non-trivial npoint/nsample
        ([1, 128, 40, 64], 2, [-2]),         # pool the other spatial axis
        ([1, 128, 40, 64], 2, [-2, -1]),     # pool both spatial axes (global)
        ([1, 16, 30], 1, [-1]),              # 1d full-extent pool
        ([1, 8, 6, 6, 10], 3, [-1]),         # 3d full-extent pool over last axis
        ([3, 40, 64], 2, [-1]),              # no batch dim
    ])
    @pytest.mark.parametrize("is_dynamic_shapes", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_max_pool_dynamic_kernel(self, input_shape, dims, axes, is_dynamic_shapes,
                                     ie_device, precision, ir_version):
        self.input_tensor = self.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(dims, axes), ie_device, precision, ir_version,
                   trace_model=True, dynamic_shapes=is_dynamic_shapes)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_max_pool_dynamic_kernel_sliding_window_unsupported(self, ie_device, precision, ir_version):
        # A static window > 1 mixed with a dynamic full-extent axis is a genuine sliding-window pool
        # that a ReduceMax cannot represent: conversion must fail with a clear message. Convert
        # directly (like the det/svd negative tests) so this expected conversion failure skips
        # _test's retry/inference plumbing.
        class aten_max_pool_mixed(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, kernel_size=[3, x.size(3)])

        example = torch.randn(1, 8, 15, 20, dtype=torch.float32)
        scripted = torch.jit.trace(aten_max_pool_mixed(), example)
        # A dynamic last axis keeps x.size(3) a runtime (non-constant) value, so the mixed
        # static-window/dynamic-extent kernel survives to the frontend and the conversion-time
        # guard fires. (A static last axis would const-fold the kernel to [3, 20] and hit the
        # ordinary static MaxPool path, which does not raise.) The translator's
        # PYTORCH_OP_CONVERSION_CHECK surfaces as OpConversionFailure; asserting the exact type
        # keeps an unrelated failure from greening this test.
        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(example,),
                             input=[ov.PartialShape([1, 8, 15, -1])])

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_max_pool_dynamic_partial_kernel_fails_at_runtime(self, ie_device, precision, ir_version):
        # A runtime (non-constant) kernel that does NOT span the full axis extent is a genuine strided
        # pool the ReduceMax decomposition cannot represent. The dynamic-kernel branch assumes a
        # full-extent global pool and inserts a runtime guard (a reshape pinning the axis to the
        # kernel value); when the real extent differs the guard must fail loudly at inference instead
        # of silently returning a wrong-shaped [N, C, H, 1] result. Here kernel = x.size(3)//2 = 32
        # while the axis extent is 64, so the guard reshape (64 -> 32) fails.
        if ie_device != "CPU":
            pytest.skip("runtime reshape-guard failure is asserted on CPU")

        class aten_max_pool_partial(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, kernel_size=[1, x.size(3) // 2])

        example = torch.randn(1, 128, 40, 64, dtype=torch.float32)
        scripted = torch.jit.trace(aten_max_pool_partial(), example)
        # Dynamic last axis keeps the kernel a runtime value, so the dynamic-kernel guard is emitted.
        ov_model = ov.convert_model(scripted, example_input=(example,),
                                    input=[ov.PartialShape([1, 128, 40, -1])])
        op_types = [n.get_type_name() for n in ov_model.get_ordered_ops()]
        assert "ReduceMax" in op_types, f"expected the dynamic-kernel branch; ops: {op_types}"
        compiled = ov.Core().compile_model(ov_model, "CPU")
        # Runtime failure in the CPU plugin (a RuntimeError, not OpConversionFailure). Assert the
        # op-labeled guard node so an unrelated failure cannot green this test.
        with pytest.raises(Exception) as exc_info:
            compiled((example.numpy(),))
        assert "require_full_extent" in str(exc_info.value)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_max_pool_dynamic_kernel_uses_reducemax(self, ie_device, precision, ir_version):
        # Structurally prove the dynamic-kernel decomposition fired: the converted OpenVINO model
        # must contain a ReduceMax and no MaxPool. The numeric tests above compare values only,
        # which a static MaxPool would also satisfy for a full-extent pool -- so they cannot, on
        # their own, distinguish the new ReduceMax branch from a const-folded static MaxPool.
        # A dynamic input shape keeps x.size(3) a runtime value so the kernel cannot const-fold.
        class aten_max_pool_dyn(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, kernel_size=[1, x.size(3)])

        example = torch.randn(1, 128, 40, 64, dtype=torch.float32)
        scripted = torch.jit.trace(aten_max_pool_dyn(), example)
        # Dynamic last axis -> the kernel stays a runtime ShapeOf value at conversion time.
        ov_model = ov.convert_model(scripted, example_input=(example,),
                                    input=[ov.PartialShape([1, 128, 40, -1])])
        op_types = [n.get_type_name() for n in ov_model.get_ordered_ops()]
        assert "ReduceMax" in op_types, f"expected the dynamic-kernel ReduceMax branch; ops: {op_types}"
        assert "MaxPool" not in op_types, f"static MaxPool must not be present; ops: {op_types}"
