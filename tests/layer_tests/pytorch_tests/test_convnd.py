# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from pytorch_layer_test_class import PytorchLayerTest


class TestConv2D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 3, 25, 25).astype(np.float32),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias):
        import torch
        import torch.nn.functional as F

        class aten_conv2d(torch.nn.Module):
            def __init__(self):
                super(aten_conv2d, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias = None
                if bias:
                    self.bias = torch.randn(weights_shape[0])
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return F.conv2d(x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.groups)

        ref_net = None

        return aten_conv2d(), ref_net, "aten::conv2d"

    @pytest.mark.parametrize("params",
                             [{'weights_shape': [1, 3, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [1, 3, 3, 3], 'strides': 2, 'pads': 0, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [1, 3, 3, 3], 'strides': 1, 'pads': 1, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [1, 3, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 2, 'groups': 1},
                              {'weights_shape': [1, 3, 3, 3], 'strides': 1, 'pads': [0, 1], 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3], 'strides': 1, 'pads': [1, 0], 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3], 'strides': 1, 'pads': 'same', 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3], 'strides': 1, 'pads': 'valid', 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [3, 1, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 1, 'groups': 3},
                              ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_conv2d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version)


class TestConv1D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 3, 25).astype(np.float32),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias):
        import torch
        import torch.nn.functional as F

        class aten_conv1d(torch.nn.Module):
            def __init__(self):
                super(aten_conv1d, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias = None
                if bias:
                    self.bias = torch.randn(weights_shape[0])
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return F.conv1d(x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.groups)

        ref_net = None

        return aten_conv1d(), ref_net, "aten::conv1d"

    @pytest.mark.parametrize("params",
                             [{'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 2, 'pads': 0, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 1, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 2, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 'same', 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 'valid', 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 1, 3], 'strides': 1, 'pads': 0, 'dilations': 1, 'groups': 3},
                              ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_conv1d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version)


class TestConv3D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 3, 25, 25, 25).astype(np.float32),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias):
        import torch
        import torch.nn.functional as F

        class aten_conv3d(torch.nn.Module):
            def __init__(self):
                super(aten_conv3d, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias = None
                if bias:
                    self.bias = torch.randn(weights_shape[0])
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return F.conv3d(x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.groups)

        ref_net = None

        return aten_conv3d(), ref_net, "aten::conv3d"

    @pytest.mark.parametrize("params",
                             [{'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 2, 'pads': 0, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': 1, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 2, 'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': [0, 1, 0], 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': [1, 0, 0], 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': [0, 0, 1], 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': [1, 1, 0], 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': [0, 1, 1], 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': [1, 0, 1], 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': 'same', 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [1, 3, 3, 3, 3], 'strides': 1, 'pads': 'valid', 'dilations': 1,
                               'groups': 1},
                              {'weights_shape': [3, 1, 3, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 1, 'groups': 3},
                              ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_conv3d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version)

class TestConv2DInSubgraph(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 3, 25, 25).astype(np.float32), np.array([1], dtype=np.int32))

    def convert_directly_via_frontend(self, model, example_input, trace_model, dynamic_shapes, ov_inputs, freeze_model):
        # Overload function to allow reproduction of issue caused by additional freeze.
        import torch

        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework('pytorch')

        model.eval()
        with torch.no_grad():
            if trace_model:
                model = torch.jit.trace(model, example_input)
            else:
                model = torch.jit.script(model)
            model = torch.jit.freeze(model)
        print(model.inlined_graph)
        decoder = TorchScriptPythonDecoder(model)
        im = fe.load(decoder)
        om = fe.convert(im)
        self._resolve_input_shape_dtype(om, ov_inputs, dynamic_shapes)
        return model, om


    def create_model(self):
        import torch
        from torchvision.ops import Conv2dNormActivation

        class aten_conv2d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                convs = []
                conv_depth=2
                for _ in range(conv_depth):
                    convs.append(Conv2dNormActivation(3, 3, 3, norm_layer=None))
                self.convs = torch.nn.Sequential(*convs)
                for layer in self.modules():
                    if isinstance(layer, torch.nn.Conv2d):
                        torch.nn.init.normal_(layer.weight)  # type: ignore[arg-type]
                        torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

            def forward(self, x, y):
                acc = self.convs(x)
                if y:
                    acc += self.convs(x)
                return acc
        ref_net = None

        return aten_conv2d(), ref_net, "aten::conv2d"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_conv2d(self, ie_device, precision, ir_version):
        self._test(*self.create_model(),
                   ie_device, precision, ir_version, freeze_model=True)
