# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from openvino.frontend import OpConversionFailure
from pytorch_layer_test_class import PytorchLayerTest


class TestAtenTo(PytorchLayerTest):
    def _prepare_input(self, input_shape=(3,)):
        return (np.random.uniform(low=0.0, high=50.0, size=input_shape),)

    def create_model(self, type, non_blocking=False, copy=False, memory_format=None):
        import torch

        class aten_to(torch.nn.Module):
            def __init__(self, type, non_blocking=False, copy=False, memory_format=None):
                super(aten_to, self).__init__()
                self.type = type
                self.non_blocking = non_blocking
                self.copy = copy
                self.memory_format = memory_format

            def forward(self, x):
                return x.to(self.type, self.non_blocking, self.copy, self.memory_format)

        ref_net = None

        return aten_to(type, non_blocking, copy, memory_format), ref_net, "aten::to"

    # Cartesian product of input/output types
    @pytest.mark.parametrize("input_type", [np.int32, np.float32, np.float64])
    @pytest.mark.parametrize("output_type",
                             [torch.uint8, torch.int8, torch.int16, torch.int32, torch.float32, torch.int64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_aten_to(self, input_type, output_type, ie_device, precision, ir_version):
        self.input_type = input_type
        self._test(*self.create_model(output_type),
                   ie_device, precision, ir_version)

    # Cartesian product of input/output types
    @pytest.mark.parametrize("input_type", [np.int32, np.float32, np.float64])
    @pytest.mark.parametrize(("output_type", "non_blocking"), [
        [torch.uint8, True],
        [torch.int8, True],
        [torch.int16, True],
        [torch.int32, True],
        [torch.int64, True],
        [torch.float32, True],
        [torch.float64, True],
        [torch.bool, True]
    ])
    @pytest.mark.nightly
    def test_aten_to_non_blocking_arg(self, input_type, output_type, non_blocking, ie_device, precision, ir_version):
        self.input_type = input_type
        self._test(*self.create_model(output_type,
                   non_blocking=non_blocking), ie_device, precision, ir_version)

    # Cartesian product of input/output types
    @pytest.mark.parametrize("input_type", [np.int32, np.float32, np.float64])
    @pytest.mark.parametrize(("output_type", "copy"), [
        [torch.uint8, True],
        [torch.int8, True],
        [torch.int16, True],
        [torch.int32, True],
        [torch.int64, True],
        [torch.float32, True],
        [torch.float64, True],
    ])
    @pytest.mark.nightly
    def test_aten_to_copy_arg(self, input_type, output_type, copy, ie_device, precision, ir_version):
        self.input_type = input_type
        self._test(*self.create_model(output_type, copy=copy),
                   ie_device, precision, ir_version)

    # Cartesian product of input/output types
    @pytest.mark.parametrize("input_type", [np.int32, np.float32, np.float64])
    @pytest.mark.parametrize(("output_type", "memory_format"), [
        [torch.uint8, 1],
        [torch.int8, 1],
        [torch.int16, 2],
        [torch.int32, 2],
        [torch.int64, 3],
        [torch.float32, 3],
        [torch.float64, 0],
    ])
    @pytest.mark.nightly
    def test_aten_to_raise_memory_format_arg(self, input_type, output_type, memory_format, ie_device, precision,
                                             ir_version):
        self.input_type = input_type
        input_shape = (3,)
        if memory_format == 2:
            input_shape = (3, 4, 5, 6)
        if memory_format == 3:
            input_shape = (3, 4, 5, 6, 7)
        self._test(*self.create_model(output_type, memory_format=memory_format), ie_device,
                   precision, ir_version, kwargs_to_prepare_input={"input_shape": input_shape})


class TestAtenToDevice(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(low=0.0, high=50.0, size=(3,)), np.random.uniform(low=0.0, high=50.0, size=(3,)))

    def create_model(self):
        import torch

        class aten_to(torch.nn.Module):

            def forward(self, x, y):
                return x.to(y.device)

        ref_net = None

        return aten_to(), ref_net, "aten::to"

    @pytest.mark.parametrize("use_trace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_aten_to_device(self, use_trace, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, trace_model=use_trace)


class TestAtenToDeviceConst(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(low=0.0, high=50.0, size=(3,)),)

    def create_model(self):
        import torch

        class aten_to(torch.nn.Module):

            def forward(self, x):
                return x.to("cpu")

        ref_net = None

        return aten_to(), ref_net, "aten::to"

    @pytest.mark.parametrize("use_trace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_aten_to_device_const(self, use_trace, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, trace_model=use_trace)


class TestAtenToComplex(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 3),)

    def create_model(self, dtype):
        import torch

        class aten_to_complex(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.view_as_real(x.to(self.dtype))

        return aten_to_complex(dtype), None, "aten::to"

    @pytest.mark.parametrize("dtype", [torch.complex32,
                                       torch.complex64,
                                       torch.complex128])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_aten_to_complex(self, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype), ie_device, precision,
                   ir_version, trace_model=True)


class TestAtenToFromComplex(PytorchLayerTest):
    def _prepare_input(self):
        # double conversion to avoid accuracy issues due to different precision
        return (np.random.randn(2, 3, 2).astype(np.float16).astype(np.float32),)

    def create_model(self, dtype):
        import torch

        class aten_to_from_complex(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                c = torch.view_as_complex(x.to(self.dtype))
                return c.to(torch.float32)

        return aten_to_from_complex(dtype), None, "aten::to"

    @pytest.mark.parametrize("dtype", [torch.float16,
                                       torch.float32,
                                       torch.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_aten_to_from_complex(self, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype), ie_device, precision,
                   ir_version)


class TestAtenToFromComplexTensor(PytorchLayerTest):
    def _prepare_input(self):
        # double conversion to avoid accuracy issues due to different precision
        return (np.random.randn(2, 3, 2).astype(np.float16).astype(np.float32),)

    def create_model(self, dtype):
        import torch

        class aten_to_from_complex(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                c = torch.view_as_complex(x.to(self.dtype))
                return c.to(x.dtype)

        return aten_to_from_complex(dtype), None, "aten::to"

    @pytest.mark.parametrize("dtype", [torch.float16,
                                       torch.float32,
                                       torch.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_aten_to_from_complex(self, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype), ie_device, precision,
                   ir_version)
