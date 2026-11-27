# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

# Random ops produce different outputs on each trace run, so the tracer always
# warns that the traced output doesn't match the Python function output.
# This is expected and not actionable at the test level.
pytestmark = pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")


class TestInplaceNormal(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (self.random.randn(1, 3, 224, 224),)

    def create_model(self, mean, std):
        class aten_normal(torch.nn.Module):
            def __init__(self, mean, std):
                super().__init__()
                self.mean = mean
                self.std = std

            def forward(self, x):
                x = x.to(torch.float32)
                return x.normal_(mean=self.mean, std=self.std), x

        return aten_normal(mean, std), "aten::normal_"

    @pytest.mark.parametrize("mean,std", [(0., 1.), (5., 20.)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_inplace_normal(self, mean, std, ie_device, precision, ir_version):
        self._test(*self.create_model(mean, std),
                   ie_device, precision, ir_version, custom_eps=1e30)


class TestNormal(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        if isinstance(self.inputs, list):
            return (self.random.randn(*self.inputs),)
        return self.inputs

    class aten_normal1(torch.nn.Module):
        def forward(self, mean, std):
            return torch.normal(mean, std)

    class aten_normal2(torch.nn.Module):
        def forward(self, mean, std):
            x = torch.empty_like(mean, dtype=torch.float32)
            return torch.normal(mean, std, out=x), x

    class aten_normal3(torch.nn.Module):
        def forward(self, mean):
            return torch.normal(mean)

    class aten_normal4(torch.nn.Module):
        def forward(self, mean):
            x = torch.empty_like(mean, dtype=torch.float32)
            return torch.normal(mean, out=x), x

    class aten_normal5(torch.nn.Module):
        def forward(self, mean):
            x = torch.empty_like(mean, dtype=torch.float32)
            return torch.normal(mean, 2., out=x), x

    class aten_normal6(torch.nn.Module):
        def forward(self, x):
            x = x.to(torch.float32)
            return torch.normal(0., 1., x.shape), x

    class aten_normal7(torch.nn.Module):
        def forward(self, x):
            x = x.to(torch.float32)
            before = x.clone()
            return torch.normal(0., 1., x.shape, out=x), x, before

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("model,inputs", [
        (aten_normal1(), (torch.arange(1., 11.).numpy(), torch.arange(1, 0, -0.1).numpy())),
        (aten_normal2(), (torch.arange(1., 11.).numpy(), torch.arange(1, 0, -0.1).numpy())),
        (aten_normal3(), (torch.arange(1., 11.).numpy(),)),
        (aten_normal4(), (torch.arange(1., 11.).numpy(),)),
        (aten_normal5(), (torch.arange(1., 11.).numpy(),)),
        (aten_normal6(), [1, 3, 224, 224]),
        (aten_normal7(), [1, 3, 224, 224]),
    ])
    @pytest.mark.precommit_torch_export
    def test_inplace_normal(self, model, inputs, ie_device, precision, ir_version):
        self.inputs = inputs
        self._test(model, "aten::normal",
                   ie_device, precision, ir_version, custom_eps=1e30)


class TestStatistics():
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("operation", ["rand_like", "randn_like"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("shape", [(), (65536,)])
    def test_export_random_like_dynamic_dtype(self, operation, dtype, shape, ie_device, precision):
        import numpy as np
        import openvino as ov
        from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder

        class Model(torch.nn.Module):
            def forward(self, x):
                return getattr(torch, operation)(x)

        program = torch.export.export(Model(), (torch.ones(shape),))
        graph = program.module()
        for node in graph.graph.nodes:
            node.meta.pop("val", None)
            node.meta.pop("tensor_meta", None)
        decoder = TorchFXPythonDecoder(graph)
        frontend = ov.frontend.FrontEndManager().load_by_framework("pytorch")
        converted = frontend.convert(frontend.load(decoder))
        parameter = converted.get_parameters()[0]
        assert parameter.get_element_type() == ov.Type.dynamic
        ov_dtype = {torch.float16: ov.Type.f16, torch.bfloat16: ov.Type.bf16, torch.float32: ov.Type.f32}[dtype]
        parameter.set_element_type(ov_dtype)
        parameter.set_partial_shape(ov.PartialShape(shape))
        converted.validate_nodes_and_infer_types()
        assert converted.output(0).get_element_type() == ov_dtype
        if dtype == torch.bfloat16:
            output = converted.get_results()[0].input_value(0)
            converted = ov.Model([ov.opset13.convert(output, ov.Type.f32)], converted.get_parameters())
            sample = ov.Tensor(ov.Type.bf16, list(shape))
            sample.data[...] = 0
        else:
            sample = torch.ones(shape, dtype=dtype).numpy()
        config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "FP32" else {}
        result = ov.Core().compile_model(converted, ie_device, config)([sample])[0]
        assert result.shape == shape
        assert result.dtype == (np.float32 if dtype == torch.bfloat16 else sample.dtype)
        assert np.isfinite(result).all()
        if operation == "rand_like":
            assert np.all((result >= 0) & (result < 1))

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("operation", ["rand", "randn"])
    @pytest.mark.parametrize("stripped_meta,expected", [(("val",), "f64"), (("val", "tensor_meta"), "f32")])
    def test_export_random_omitted_dtype(self, operation, stripped_meta, expected, ie_device, precision):
        import numpy as np
        import openvino as ov
        from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder

        class Model(torch.nn.Module):
            def forward(self):
                return getattr(torch, operation)(2, 3)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            graph = torch.export.export(Model(), ()).module()
        finally:
            torch.set_default_dtype(default_dtype)
        for node in graph.graph.nodes:
            assert "dtype" not in node.kwargs
            for key in stripped_meta:
                node.meta.pop(key, None)
        decoder = TorchFXPythonDecoder(graph)
        frontend = ov.frontend.FrontEndManager().load_by_framework("pytorch")
        converted = frontend.convert(frontend.load(decoder))
        assert converted.output(0).get_element_type() == getattr(ov.Type, expected)
        config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "FP32" else {}
        result = ov.Core().compile_model(converted, ie_device, config)()[0]
        assert result.shape == (2, 3)
        assert np.isfinite(result).all()

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("operation", ["rand", "randn", "rand_like", "randn_like"])
    def test_export_random_scalar(self, operation, ie_device, precision):
        import numpy as np
        import openvino as ov

        class Model(torch.nn.Module):
            def forward(self, x):
                if operation.endswith("_like"):
                    return getattr(torch, operation)(x)
                return getattr(torch, operation)(x.shape)

        example = torch.tensor(1.0)
        program = torch.export.export(Model(), (example,))
        converted = ov.convert_model(program)
        assert converted.output(0).get_partial_shape() == ov.PartialShape([])
        config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "FP32" else {}
        compiled = ov.Core().compile_model(converted, ie_device, config)
        result = compiled([example.numpy()])[0] if converted.inputs else compiled()[0]
        assert result.shape == ()
        assert np.isfinite(result)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("operation", ["rand", "randn"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
    def test_export_random_out(self, operation, dtype, ie_device, precision):
        import numpy as np
        import openvino as ov

        class Model(torch.nn.Module):
            def forward(self, x):
                before = x + 1
                result = getattr(torch, operation)(x.shape, out=x)
                return result, x + result, before

        example = torch.zeros(3, 256, dtype=dtype)
        dynamic = ({0: torch.export.Dim("batch")},) if operation == "rand" else None
        program = torch.export.export(Model(), (example,), dynamic_shapes=dynamic)
        converted = ov.convert_model(program)
        config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "FP32" else {}
        compiled = ov.Core().compile_model(converted, ie_device, config)
        sample = torch.zeros(5 if dynamic else 3, 256, dtype=dtype)
        outputs = compiled([sample.numpy()])
        first, second = outputs[0], outputs[1]
        assert first.shape == tuple(sample.shape)
        assert first.dtype == sample.numpy().dtype
        assert np.isfinite(first).all()
        np.testing.assert_allclose(second, first * 2, rtol=1e-3, atol=1e-3)
        np.testing.assert_array_equal(outputs[2], np.ones_like(sample.numpy()))

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("operation", ["rand_like", "randn_like"])
    def test_export_random_like_out(self, operation, ie_device, precision):
        import numpy as np
        import openvino as ov

        class Model(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                getattr(torch.ops.aten, operation).out(x, out=out)
                return out, x + out

        example = torch.zeros(3, 256)
        program = torch.export.export(Model(), (example,))
        converted = ov.convert_model(program)
        config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "FP32" else {}
        outputs = ov.Core().compile_model(converted, ie_device, config)([example.numpy()])
        first, second = outputs[0], outputs[1]
        assert first.shape == tuple(example.shape)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(second, first)
        if operation == "rand_like":
            assert np.all((first >= 0) & (first < 1))

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("operation", ["rand", "randn"])
    def test_export_random_default_dtype(self, operation, ie_device, precision):
        import numpy as np
        import openvino as ov

        class Model(torch.nn.Module):
            def forward(self, x):
                return getattr(torch, operation)(x.shape)

        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            program = torch.export.export(Model(), (torch.ones(3, 5, dtype=torch.float32),))
        finally:
            torch.set_default_dtype(previous_dtype)
        converted = ov.convert_model(program)
        assert converted.output(0).get_element_type() == ov.Type.f64
        config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "FP32" else {}
        result = ov.Core().compile_model(converted, ie_device, config)()[0]
        assert result.dtype == np.float64
        assert np.isfinite(result).all()

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("operation", ["rand", "randn", "rand_like", "randn_like"])
    @pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16, torch.float64])
    def test_export_random(self, operation, dtype, ie_device, precision):
        import numpy as np
        import openvino as ov

        class Model(torch.nn.Module):
            def forward(self, x):
                if operation.endswith("_like"):
                    return getattr(torch, operation)(x, dtype=dtype)
                return getattr(torch, operation)(x.shape, dtype=dtype)

        model = Model().eval()
        example = torch.ones(5, 4096, dtype=torch.float64)
        program = torch.export.export(model, (example,), dynamic_shapes=({0: torch.export.Dim("batch")},))
        assert any(str(node.target) == f"aten.{operation}.default" for node in program.graph.nodes)
        converted = ov.convert_model(program)
        reference_dtype = model(example).dtype
        if reference_dtype == torch.bfloat16:
            assert converted.output(0).get_element_type() == ov.Type.bf16
            output = converted.get_results()[0].input_value(0)
            converted = ov.Model([ov.opset13.convert(output, ov.Type.f32)], converted.get_parameters())
        config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "FP32" else {}
        compiled = ov.Core().compile_model(converted, ie_device, config)

        for batch in (2, 9):
            sample = torch.ones(batch, 4096, dtype=torch.float64)
            result = compiled([sample.numpy()])[0]
            assert result.shape == tuple(sample.shape)
            assert result.dtype == (np.float32 if reference_dtype == torch.bfloat16 else model(sample).numpy().dtype)
            assert np.isfinite(result).all()
            assert np.ptp(result) > 0
            if operation in ("rand", "rand_like"):
                assert np.all((result >= 0) & (result < 1))

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("shape", [(257,), (16, 32)])
    @pytest.mark.parametrize("use_out", [False, True])
    def test_normal_scalar_mean(self, shape, use_out, ie_device, precision):
        import numpy as np
        import openvino as ov

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.use_out = use_out

            def forward(self, std):
                if self.use_out:
                    out = torch.empty_like(std)
                    return torch.normal(2.5, std, out=out)
                return torch.normal(2.5, std)

        std = torch.full(shape, 3.0)
        model = Model()
        if PytorchLayerTest.use_torch_export():
            input_model = torch.export.export(model, (std,))
        else:
            input_model = torch.jit.trace(model, (std,), check_trace=False)
        ov_model = ov.convert_model(input_model)
        config = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "FP32" else {}
        compiled_model = ov.Core().compile_model(ov_model, ie_device, config)
        result = compiled_model([std.numpy()])[0]

        assert result.shape == shape
        assert np.isfinite(result).all()
        # A scalar random sample broadcast over std has the right shape but no
        # variation between elements. This must generate a sample per element.
        assert np.ptp(result) > 0

    class aten_normal(torch.nn.Module):
        def forward(self, mean, std):
            return torch.normal(mean, std)

    class aten_randn(torch.nn.Module):
        def forward(self, size):
            return torch.randn(*size)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("fw_model,inputs", [
        (aten_normal(), (0, 1, (1000000,))),
        (aten_normal(), (0, 1, (10000, 100))),
        (aten_normal(), (0, 3, (100000, 100))),
        (aten_normal(), (1, 6, (100000, 100))),
        (aten_normal(), (-20, 2, (10000, 100))),
        (aten_normal(), (-20, 100, (10000, 100))),

        (aten_randn(), (0, 1, (1000000,))),
        (aten_randn(), (0, 1, (10000, 100))),
        (aten_randn(), (0, 1, (100000, 100))),
    ])
    def test_normal_statistics(self, fw_model, inputs, ie_device, precision):
        import numpy.testing as npt
        import numpy as np
        import openvino as ov
        mean_scalar, std_scalar, size = inputs
        mean = torch.full(size, mean_scalar, dtype=torch.float32)
        std = torch.full(size, std_scalar, dtype=torch.float32)

        if isinstance(fw_model, self.aten_randn):
            example_input = (torch.tensor(size), )
            input_size = [len(size)]
        else:
            example_input = (mean, std)
            input_size = [size, size]

        ov_model = ov.convert_model(input_model=fw_model, example_input=example_input, input=input_size)
        if ie_device == 'GPU' and precision == 'FP32':
            config = {'INFERENCE_PRECISION_HINT': 'f32'}
        else:
            config = {}
        compiled_model = ov.Core().compile_model(ov_model, ie_device, config)

        fw_res = fw_model(*example_input)
        ov_res = compiled_model(example_input)[0]

        x_min, x_max = mean_scalar - 2 * std_scalar, mean_scalar + 2 * std_scalar
        hist_fw, _ = np.histogram(fw_res.numpy(), bins=100, range=(x_min, x_max))
        hist_ov, _ = np.histogram(ov_res, bins=100, range=(x_min, x_max))
        npt.assert_allclose(hist_fw, hist_ov, atol=0.2, rtol=0.2)
