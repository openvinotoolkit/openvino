# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestRound(PytorchLayerTest):
    def _prepare_input(self, out=False, dtype="float32"):
        import numpy as np
        input = self.random.randn(1, 3, 224, 224, dtype=dtype)
        if dtype == "float64":
            # fp64 can fail by accuracy, because pytorch rounds fp64 value and ov will round fp32 value.
            # To remove sporadic accuracy fails we will round the number to 6 decimal places.
            input = np.round(input, 6)
        if not out:
            return (input, )
        return (input, np.zeros_like(input))

    def create_model(self, out=False):
        import torch

        class aten_round(torch.nn.Module):
            def __init__(self, out):
                super().__init__()
                if out:
                    self.forward = self.forward_out


            def forward(self, x):
                return torch.round(x)

            def forward_out(self, x, y):
                return torch.round(x, out=y), y


        return aten_round(out), "aten::round"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
    def test_round(self, out, dtype, ie_device, precision, ir_version):
        if ie_device == "GPU" and dtype not in ["float32", "float64"]:
            pytest.xfail(reason="square is not supported on GPU for integer types")
        self._test(*self.create_model(out), ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "dtype": dtype})


class TestRoundScalar(PytorchLayerTest):
    def _prepare_input_int(self):
        import numpy as np
        return (np.array(self.random.randint(low=-5, high=5)), )

    def _prepare_input_float(self):
        import numpy as np
        return (np.array(self.random.uniform(low=-5, high=5)), )

    def create_model(self, input_type="float"):
        import torch

        class aten_round(torch.nn.Module):
            def __init__(self, input_type):
                super().__init__()
                if input_type == "int":
                    self.forward = self.forward_int
                else:
                    self.forward = self.forward_float


            def forward_int(self, x:int):
                return torch.round(x)

            def forward_float(self, x:float):
                return torch.round(x)


        return aten_round(input_type), "aten::round"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_type", ["int", "float"])
    def test_round(self, input_type, ie_device, precision, ir_version):
        if input_type == "int":
            if ie_device == "GPU":
                pytest.xfail(reason="round is not supported on GPU for integer types")
            self._prepare_input = self._prepare_input_int
        else:
            self._prepare_input = self._prepare_input_float
        self._test(*self.create_model(input_type), ie_device, precision, ir_version, trace_model=True)


class TestRoundDecimals(PytorchLayerTest):
    # torch.round(x, decimals=...) was previously converted as a plain integer round: the
    # decimals value (TorchScript scalar input / FX aten.round.decimals kwarg) was ignored.
    def _prepare_input(self, dtype="float32"):
        import numpy as np
        # includes half-way ties and negatives to exercise HALF_TO_EVEN and sign handling.
        # 6500 with decimals=-3 is 6.5 after the division, so it also catches a pre-round Divide
        # that CommonOptimizations would rewrite into a reciprocal multiplication.
        input = np.array([1.234, 2.567, -1.256, 0.05, 3.449, 0.5, 1.5, 2.5,
                          -0.5, -1.5, 6.5, 1234.0, 1250.0, 2.675, 6500.0], dtype=dtype)
        return (input,)

    def create_model(self, decimals):
        import torch

        class aten_round_decimals(torch.nn.Module):
            def __init__(self, decimals):
                super().__init__()
                self.decimals = decimals

            def forward(self, x):
                return torch.round(x, decimals=self.decimals)

        return aten_round_decimals(decimals), "aten::round"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("decimals", [1, 2, -1, -3])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_round_decimals(self, decimals, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(decimals), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": dtype})


class TestRoundDecimalsLowPrecision(PytorchLayerTest):
    # torch evaluates the rounding kernel in opmath_t, so half and bfloat16 inputs are widened to
    # float before the scaling with 10^decimals. Scaling them in the input type instead overflows
    # (f16 1000 with decimals=2 became inf instead of staying 1000) and loses the digits that the
    # rounding is supposed to keep.
    def _prepare_input(self):
        import numpy as np
        # 65504 is the largest finite float16: scaling it (or 1000 with decimals=2) in float16
        # overflows instead of rounding, which is what the widening to float protects against.
        # The small values stay below the rounding threshold of the negative decimals, so they
        # have to come back as zeros.
        input = np.array([1000.0, 2000.0, 1234.0, 65504.0, -1000.0, 0.5, 1.5, 2.5, -2.5, 1e-4],
                         dtype=np.float32)
        return (input,)

    def create_model(self, decimals, torch_dtype):
        import torch

        class aten_round_decimals(torch.nn.Module):
            def __init__(self, decimals, torch_dtype):
                super().__init__()
                self.decimals = decimals
                self.torch_dtype = torch_dtype

            def forward(self, x):
                return torch.round(x.to(self.torch_dtype), decimals=self.decimals).to(torch.float32)

        return aten_round_decimals(decimals, torch_dtype), "aten::round"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("decimals", [1, 2, -1, -8])
    @pytest.mark.parametrize("torch_dtype", ["float16", "bfloat16"])
    def test_round_decimals_low_precision(self, decimals, torch_dtype, ie_device, precision, ir_version):
        import torch

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[torch_dtype]
        self._test(*self.create_model(decimals, torch_dtype), ie_device, precision, ir_version)


class TestRoundDecimalsDirectLowPrecision(PytorchLayerTest):
    # The input is already f16, without the internal .to() of the test above: the direct frontend
    # conversion path leaves the input element type dynamic during translation and specializes it
    # only afterwards, so the widening to float must not depend on the type being statically known
    # (otherwise f16 1000 with decimals=2 scales to inf instead of staying 1000).
    def _prepare_input(self):
        import numpy as np
        input = np.array([1000.0, 2000.0, 1234.0, -1000.0, 0.5, 1.5, 2.5, -2.5], dtype=np.float16)
        return (input,)

    def create_model(self, decimals):
        import torch

        class aten_round_decimals(torch.nn.Module):
            def __init__(self, decimals):
                super().__init__()
                self.decimals = decimals

            def forward(self, x):
                return torch.round(x, decimals=self.decimals)

        return aten_round_decimals(decimals), "aten::round"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("decimals", [1, 2, -1])
    def test_round_decimals_direct_low_precision(self, decimals, ie_device, precision, ir_version):
        self._test(*self.create_model(decimals), ie_device, precision, ir_version)


@pytest.mark.nightly
@pytest.mark.precommit
def test_round_decimals_dynamic_f64_working_type(ie_device, precision, ir_version):
    # The direct frontend path leaves f64 inputs dynamic during translation. Widening every
    # dynamic input to f32 would lose precision before ConvertLike restores the f64 output type.
    import torch
    from openvino import PartialShape, Type
    from openvino.frontend import FrontEndManager
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    class aten_round_decimals(torch.nn.Module):
        def forward(self, x):
            return torch.round(x, decimals=8)

    decoder = TorchScriptPythonDecoder(aten_round_decimals())
    fe = FrontEndManager().load_by_framework("pytorch")
    model = fe.convert(fe.load(decoder))
    parameter = model.inputs[0].get_node()
    assert parameter.get_element_type().is_dynamic()
    parameter.set_element_type(Type.f64)
    parameter.set_partial_shape(PartialShape([2]))
    model.validate_nodes_and_infer_types()

    promotions = [node for node in model.get_ordered_ops() if node.get_type_name() == "ConvertPromoteTypes"]
    assert len(promotions) == 1
    assert all(output.get_element_type() == Type.f64 for output in promotions[0].outputs())


class TestRoundDecimalsOut(PytorchLayerTest):
    # aten::round(self, decimals, out) passes data, decimals and out as three inputs, which the
    # translator rejected before parsing; the decimals and out combination is a valid overload.
    def _prepare_input(self, dtype="float32"):
        import numpy as np
        input = np.array([1.234, 2.567, -1.256, 0.05, 3.449, 0.5, 1.5, 2.5,
                          -0.5, -1.5, 6.5, 1234.0, 1250.0, 2.675], dtype=dtype)
        return (input, np.zeros_like(input))

    def create_model(self, decimals):
        import torch

        class aten_round_decimals_out(torch.nn.Module):
            def __init__(self, decimals):
                super().__init__()
                self.decimals = decimals

            def forward(self, x, y):
                return torch.round(x, decimals=self.decimals, out=y), y

        return aten_round_decimals_out(decimals), "aten::round"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("decimals", [1, 2, -1])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_round_decimals_out(self, decimals, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(decimals), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": dtype})
