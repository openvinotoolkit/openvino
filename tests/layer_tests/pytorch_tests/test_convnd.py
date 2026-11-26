# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from pytorch_layer_test_class import PytorchLayerTest


class TestConv2D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (self.random.randn(2, 3, 25, 25),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias):
        import torch
        import torch.nn.functional as F

        class aten_conv2d(torch.nn.Module):
            def __init__(self, rng):
                super().__init__()
                self.weight = rng.torch_randn(*weights_shape)
                self.bias = None
                if bias:
                    self.bias = rng.torch_randn(weights_shape[0])
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return F.conv2d(x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.groups)


        return aten_conv2d(self.random), "aten::conv2d"

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
    @pytest.mark.precommit_torch_export
    def test_conv2d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version)


class TestConv1D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (self.random.randn(2, 3, 25),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias):
        import torch
        import torch.nn.functional as F

        class aten_conv1d(torch.nn.Module):
            def __init__(self, rng):
                super().__init__()
                self.weight = rng.torch_randn(*weights_shape)
                self.bias = None
                if bias:
                    self.bias = rng.torch_randn(weights_shape[0])
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return F.conv1d(x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.groups)


        return aten_conv1d(self.random), "aten::conv1d"

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
    @pytest.mark.precommit_torch_export
    def test_conv1d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version)


class TestConv3D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (self.random.randn(2, 3, 25, 25, 25),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias):
        import torch
        import torch.nn.functional as F

        class aten_conv3d(torch.nn.Module):
            def __init__(self, rng):
                super().__init__()
                self.weight = rng.torch_randn(*weights_shape)
                self.bias = None
                if bias:
                    self.bias = rng.torch_randn(weights_shape[0])
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return F.conv3d(x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.groups)


        return aten_conv3d(self.random), "aten::conv3d"

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
    @pytest.mark.precommit_torch_export
    def test_conv3d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version)

class TestPadConv2DSame(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(1, 3, 24, 30),)

    def create_model(self, kernel, dilation):
        import torch
        import torch.nn.functional as F

        class pad_conv2d_same(torch.nn.Module):
            def __init__(self, rng):
                super().__init__()
                self.weight = rng.torch_randn(4, 3, kernel, kernel)
                self.dilation = dilation

            def forward(self, x):
                return F.conv2d(F.pad(x, [0, 1, 0, 1]), self.weight, padding="same", dilation=self.dilation)

        return pad_conv2d_same(self.random), "aten::conv2d"

    @pytest.mark.parametrize("kernel,dilation", [(3, 1), (4, 1), (3, 2)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_pad_conv2d_same(self, kernel, dilation, ie_device, precision, ir_version):
        self._test(*self.create_model(kernel, dilation), ie_device, precision, ir_version)


@pytest.mark.parametrize("capture", ["trace", "export"])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("kernel_parity", ["odd", "even"])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.precommit
def test_conv2d_same_dynamic_kernel_after_pad(capture, groups, kernel_parity, dilation, ie_device, precision):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import openvino as ov

    class Model(torch.nn.Module):
        def forward(self, x, weight):
            return F.conv2d(
                F.pad(x, [0, 1, 0, 1]),
                weight,
                padding="same",
                dilation=dilation,
                groups=groups,
            )

    model = Model().eval()
    x = torch.randn(1, 4, 12, 14)
    sample_kernel = 3 if kernel_parity == "odd" else 4
    weight = torch.randn(4, 4 // groups, sample_kernel, sample_kernel)
    if capture == "trace":
        traced = torch.jit.trace(model, (x, weight))
        converted = ov.convert_model(
            traced,
            input=[ov.PartialShape(x.shape), ov.PartialShape([4, 4 // groups, -1, -1])],
        )
    else:
        if kernel_parity == "odd":
            kernel_h = 2 * torch.export.Dim("kernel_h_half", min=2, max=3) - 1
            kernel_w = 2 * torch.export.Dim("kernel_w_half", min=2, max=3) - 1
            kernel_shapes = ((3, 3), (3, 5), (5, 5))
        else:
            kernel_h = 2 * torch.export.Dim("kernel_h_half", min=1, max=3)
            kernel_w = 2 * torch.export.Dim("kernel_w_half", min=1, max=3)
            kernel_shapes = ((2, 2), (2, 4), (4, 4))
        exported = torch.export.export(
            model,
            (x, weight),
            dynamic_shapes=({}, {2: kernel_h, 3: kernel_w}),
        )
        converted = ov.convert_model(exported)
    if capture == "trace":
        kernel_shapes = ((2, 2), (2, 4), (4, 4)) if kernel_parity == "even" else ((3, 3), (3, 5), (5, 5))

    for kernel_shape in kernel_shapes:
        shape_model = converted.clone()
        weight_shape = [4, 4 // groups, *kernel_shape]
        shape_model.reshape(dict(zip(shape_model.inputs, [x.shape, weight_shape])))
        config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "FP32" else {}
        compiled = ov.Core().compile_model(shape_model, ie_device, config)
        test_x = torch.randn_like(x)
        test_weight = torch.randn(weight_shape)
        actual = compiled([test_x.numpy(), test_weight.numpy()])[0]
        expected = model(test_x, test_weight).numpy()
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


class TestConv2DInSubgraph(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (self.random.randn(2, 3, 25, 25), np.array([1], dtype=np.int32))

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

        return aten_conv2d(), "aten::conv2d"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_conv2d(self, ie_device, precision, ir_version):
        self._test(*self.create_model(),
                   ie_device, precision, ir_version, freeze_model=True)
