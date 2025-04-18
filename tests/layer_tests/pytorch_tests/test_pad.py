# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestPad(PytorchLayerTest):
    def _prepare_input(self, ndim=4, dtype="float32"):
        import numpy as np
        input_5d_shape = [1, 3, 14, 14, 18]
        return (np.random.randn(*input_5d_shape[:ndim]).astype(dtype),)

    def create_model(self, pads, mode, value=None):
        import torch
        import torch.nn.functional as F

        class aten_pad(torch.nn.Module):
            def __init__(self, pads, mode, value=None):
                super().__init__()
                self.pads = pads
                self.mode = mode
                self.value = value

            def forward(self, x):
                return F.pad(x, self.pads, mode=self.mode, value=self.value)

        ref_net = None

        return aten_pad(pads, mode, value), ref_net, "aten::pad"

    @pytest.mark.parametrize("pads,mode,value,dtype", [
        ((1, 2, 3, 4), "reflect", None, "float32"),
        ((1, 0, 0, 0, 0, 1), "reflect", None, "float64"),
        ((0, 0, 0, 0, 0, 0), "reflect", None, "float32"),
        ((1, 2, 3, 4), "replicate", None, "float32"),
        ((1, 0, 0, 0, 0, 0), "replicate", None, "float64"),
        ((1, 0, 0, 0, 0, 1), "replicate", None, "float32"),
        ((0, 0, 0, 0, 0, 0), "replicate", None, "float64"),
        ((1, 2, 3, 4), "constant", None, "int32"),
        ((1, 2, 3, 4), "constant", 42., "int64"),
        ((1, 2, 3, 4), "constant", -0.57, "float32"),
        ((1, 2), "constant", None, "float64"),
        ((1, 0, 0, 0, 0, 1), "constant", None, "int8"),
        ((0, 0, 0, 0, 0, 0), "constant", None, "int64"),
        ((1, 0, 0, 0, 0, 1, 1, 2), "constant", 0., "int32"),
        ((1, 2, 0, 0), "circular", None, "int32"),
        ((1, 2, 3, 4), "circular", None, "int64"),
        ((0, 1, 0, 0), "circular", None, "float32"),
        ((0, 0, 0, 0), "circular", None, "float64"),
        ((0, 0, -1, -2), "circular", None, "int8"),
        ((-1, -2, -1, -2), "circular", None, "float32"),
        ((-5, -8, 0, 0), "circular", None, "float32"),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pad4d(self, pads, mode, value, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(pads, mode, value), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"ndim": 4, "dtype": dtype})

    @pytest.mark.parametrize("pads,mode,value,dtype", [
        ((1, 2, 3, 4, 5, 6), "reflect", None, "float32"),
        ((1, 0, 0, 0, 0, 1), "reflect", None, "float64"),
        ((1, 0, 0, 0, 0, 0), "reflect", None, "float32"),
        ((0, 0, 0, 0, 0, 0), "reflect", None, "float64"),
        ((1, 2, 3, 4, 5, 6), "replicate", None, "float32"),
        ((1, 0, 0, 0, 0, 0), "replicate", None, "float64"),
        ((1, 0, 0, 0, 0, 1), "replicate", None, "float32"),
        ((0, 0, 0, 0, 0, 0), "replicate", None, "float64"),
        ((1, 2, 3, 4), "constant", None, "int32"),
        ((1, 2, 3, 4), "constant", 42., "float32"),
        ((1, 2, 3, 4), "constant", -0.57, "float64"),
        ((1, 2), "constant", None, "int64"),
        ((1, 0, 0, 0, 0, 1), "constant", None, "float32"),
        ((0, 0, 0, 0, 0, 0), "constant", None, "float64"),
        ((1, 0, 0, 0, 0, 1, 1, 2), "constant", 0., "int8"),
        ((1, 0, 0, 0, 0, 1, 1, 2, 2, 3), "constant", 0., "float32"),
        ((1, 2, 0, 0, 0, 0), "circular", None, "float32"),
        ((1, 2, 3, 4, 5, 6), "circular", None, "float64"),
        ((0, 1, 0, 0, 0, 0), "circular", None, "int32"),
        ((0, 0, 0, 0, 0, 0), "circular", None, "int64"),
        ((0, 0, -1, -2, 0, 0), "circular", None, "int8"),
        ((-1, -2, -1, -2, -1, -2), "circular", None, "float32"),
        ((-5, -8, 0, 0, 0, 0), "circular", None, "float32"),
        ((10, 10, 10, 10, 10, 10), "circular", None, "float32"),
    ])
    @pytest.mark.nightly
    def test_pad5d(self, pads, mode, value, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(pads, mode, value), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"ndim": 5, "dtype": dtype}, trace_model=True)

    @pytest.mark.parametrize("pads,mode,value,dtype", [
        ((1, 2), "reflect", None, 'float32'),
        ((1, 0), "reflect", None, "float64"),
        ((0, 0), "reflect", None, "float32"),
        ((1, 2), "replicate", None, "float32"),
        ((1, 0), "replicate", None, "float64"),
        ((0, 0), "replicate", None, "float64"),
        ((1, 0), "constant", None, "int32"),
        ((1, 0), "constant", 42., "float32"),
        ((1, 0), "constant", -0.57, "float64"),
        ((1, 2, 3, 4), "constant", None, "float64"),
        ((1, 2, 3, 4), "constant", 42., "int64"),
        ((1, 2, 3, 4), "constant", -0.57, "float32"),
    ])
    @pytest.mark.nightly
    def test_pad2d(self, pads, mode, value, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(pads, mode, value), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"ndim": 2, "dtype": dtype}, trace_model=True)


class TestPadListPaddingings(PytorchLayerTest):
    def _prepare_input(self, ndim=4, pad_w=0, pad_h=0, dtype="float32"):
        import numpy as np
        input_5d_shape = [1, 3, 14, 14, 18]
        return (np.random.randn(*input_5d_shape[:ndim]).astype(dtype), np.array(pad_w, dtype=np.int32), np.array(pad_h, dtype=np.int32))

    def create_model(self, mode, value=None):
        import torch
        import torch.nn.functional as F

        class aten_pad(torch.nn.Module):
            def __init__(self, mode, value=None):
                super().__init__()
                self.mode = mode
                self.value = value

            def forward(self, x, pad_w: int, pad_h: int):
                return F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=self.value)

        ref_net = None

        return aten_pad(mode, value), ref_net, "aten::pad"

    @pytest.mark.parametrize("pad_w,pad_h,mode,value,dtype", [
        (2, 0, "reflect", None, "float32"),
        (0, 2, "reflect", None, "float64"),
        (10, 10, "reflect", None, "float32"),
        (0, 0, "reflect", None, "float32"),
        (5, 3, "reflect", None, "float64"),
        (2, 0, "replicate", None, "float32"),
        (0, 2, "replicate", None, "float64"),
        (10, 10, "replicate", None, "float32"),
        (5, 3, "replicate", None, "float64"),
        (0, 0, "replicate", None, "float32"),
        (2, 0, "constant", None, "int32"),
        (0, 3, "constant", 42., "int64"),
        (4, 4, "constant", -0.57, "float32"),
        (1, 2, "constant", None, "int8"),
        (0, 0, "constant", -0.57, "float64"),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pad4d(self, pad_w, pad_h, mode, value, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(mode, value), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"ndim": 4, "pad_w": pad_w, "pad_h": pad_h, "dtype": dtype})

    @pytest.mark.parametrize("pad_w,pad_h,mode,value", [
        (2, 0, "reflect", None),
        (0, 2, "reflect", None),
        (10, 10, "reflect", None),
        (0, 0, "reflect", None),
        (5, 3, "reflect", None),
        (2, 0, "replicate", None),
        (0, 2, "replicate", None),
        (10, 10, "replicate", None),
        (5, 3, "replicate", None),
        (0, 0, "replicate", None),
        (2, 0, "constant", None),
        (0, 3, "constant", 42.),
        (4, 4, "constant", -0.57),
        (1, 2, "constant", None),
        (0, 0, "constant", -0.57)
    ])
    @pytest.mark.nightly
    def test_pad5d(self, pad_w, pad_h, mode, value, ie_device, precision, ir_version):
        self._test(*self.create_model(mode, value), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"ndim": 5, "pad_w": pad_w, "pad_h": pad_h})

    @pytest.mark.parametrize("pad_w,pad_h,mode,value,dtype", [
        (2, 0, "reflect", None, "float32"),
        (0, 2, "reflect", None, "float64"),
        (10, 10, "reflect", None, "float64"),
        (0, 0, "reflect", None, "float32"),
        (5, 3, "reflect", None, "float64"),
        (2, 0, "replicate", None, "float32"),
        (0, 2, "replicate", None, "float64"),
        (10, 10, "replicate", None, "float32"),
        (5, 3, "replicate", None, "float64"),
        (0, 0, "replicate", None, "float32"),
        (2, 0, "constant", None, "int64"),
        (0, 3, "constant", 42., "int32"),
        (4, 4, "constant", -0.57, "float32"),
        (1, 2, "constant", None, "int8"),
        (0, 0, "constant", -0.57, "float64")
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pad2d(self, pad_w, pad_h, mode, value, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(mode, value), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"ndim": 2, "pad_w": pad_w, "pad_h": pad_h, "dtype": dtype})


class TestReflectionPad(PytorchLayerTest):
    def _prepare_input(self, ndim=4, dtype="float32"):
        import numpy as np
        input_5d_shape = [1, 3, 14, 14, 18]
        return (np.random.randn(*input_5d_shape[:ndim]).astype(dtype),)
    
    class aten_pad(torch.nn.Module):
        def __init__(self, pads, ndim):
            super().__init__()
            self.pads = pads
            self.ndim = ndim

        def forward(self, x):
            if(self.ndim == 1):
                return torch.ops.aten.reflection_pad1d(x, self.pads)
            elif(self.ndim == 2):
                return torch.ops.aten.reflection_pad2d(x, self.pads)
            elif(self.ndim == 3):
                return torch.ops.aten.reflection_pad3d(x, self.pads)
            else:
                raise Exception("Unsupported pads")

    def create_model(self, pads):
        ndim = len(pads) / 2
        if(ndim == 1):
            return self.aten_pad(pads, ndim), None, "aten::reflection_pad1d"
        elif(ndim == 2):
            return self.aten_pad(pads, ndim), None, "aten::reflection_pad2d"
        elif(ndim == 3):
            return self.aten_pad(pads, ndim), None, "aten::reflection_pad3d"
        else:
            raise Exception("Unsupported pads")

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
    @pytest.mark.parametrize("pads", [
        (1, 2),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 3, 2),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_reflection_padnd(self, pads, dtype, ie_device, precision, ir_version):
        ndim = len(pads) // 2 + 2
        print(ndim)
        self._test(*self.create_model(pads), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"ndim": ndim, "dtype": dtype})

class TestReplicatePad1D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.rand(*self.input_shape).astype(self.dtype),)

    def create_model(self, pads):
        import torch
        import torch.nn.functional as F

        class aten_pad(torch.nn.Module):
            def __init__(self, pads):
                super().__init__()
                self.pads = pads

            def forward(self, x):
                return torch.ops.aten.replication_pad1d(x,self.pads)
            
        return aten_pad(pads), None, "aten::replication_pad1d"

    @pytest.mark.parametrize("input_shape", [(1,2,3),(3,4,5),(5,6,7)])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
    @pytest.mark.parametrize("pads", [
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 4),
    ])
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_replicate_padnd(self, pads, input_shape, dtype, ie_device, precision, ir_version):
        self.input_shape = input_shape
        self.dtype = dtype
        self._test(*self.create_model(pads), ie_device, precision, ir_version)

class TestReplicatePad2D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.rand(*self.input_shape).astype(self.dtype),)

    def create_model(self, pads):
        import torch
        import torch.nn.functional as F

        class aten_pad(torch.nn.Module):
            def __init__(self, pads):
                super().__init__()
                self.pads = pads

            def forward(self, x):
                return torch.ops.aten.replication_pad2d(x,self.pads)
            
        return aten_pad(pads), None, "aten::replication_pad2d"
    @pytest.mark.parametrize("input_shape", [(1,2,3,4),(2,3,4,5),(3,4,5,6)])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
    @pytest.mark.parametrize("pads", [
        (1, 2, 2, 1),
        (2, 1, 3, 4),
        (2, 3, 1, 2),
        (3, 4, 5, 6),
    ])
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_replicate_padnd(self, pads, input_shape, dtype, ie_device, precision, ir_version):
        self.input_shape = input_shape
        self.dtype = dtype
        self._test(*self.create_model(pads), ie_device, precision, ir_version)
    
class TestReplicatePad3D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.rand(*self.input_shape).astype(self.dtype),)

    def create_model(self, pads):
        import torch
        import torch.nn.functional as F

        class aten_pad(torch.nn.Module):
            def __init__(self, pads):
                super().__init__()
                self.pads = pads

            def forward(self, x):
                return torch.ops.aten.replication_pad3d(x,self.pads)
            
        return aten_pad(pads), None, "aten::replication_pad3d"

    @pytest.mark.parametrize("input_shape", [(1,2,3,4,5),(2,3,4,5,6),(3,4,5,6,7)])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
    @pytest.mark.parametrize("pads", [
        (1, 2, 2, 1, 3, 4),
        (2, 1, 3, 4, 2, 1),
        (2, 3, 1, 2, 2, 1),
        (3, 4, 5, 6, 1, 2),
    ])
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_replicate_padnd(self, pads, input_shape, dtype, ie_device, precision, ir_version):
        self.input_shape = input_shape
        self.dtype = dtype
        self._test(*self.create_model(pads), ie_device, precision, ir_version)
    