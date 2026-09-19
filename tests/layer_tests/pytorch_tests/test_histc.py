# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import openvino as ov

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestHistc(PytorchLayerTest):
    def _prepare_input(self, dtype="float32", shape=(20,), low=-10.0, high=10.0, bins=10, out=False, fixed=None):
        if fixed is not None:
            data = np.array(fixed, dtype=dtype)
        else:
            data = self.random.uniform(low, high, size=shape).astype(dtype)
        if not out:
            return (data,)
        return (data, np.zeros((bins,), dtype=dtype))

    def create_model(self, bins, min, max, out=False, omit=None):
        class aten_histc(torch.nn.Module):
            def __init__(self, bins, min, max, out, omit):
                super().__init__()
                self.bins = bins
                self.min = min
                self.max = max
                if omit == "all":
                    self.forward = self.forward_no_args
                elif omit == "range":
                    self.forward = self.forward_bins_only
                elif out:
                    self.forward = self.forward_out

            def forward(self, x):
                return torch.histc(x, bins=self.bins, min=self.min, max=self.max)

            def forward_no_args(self, x):
                return torch.histc(x)

            def forward_bins_only(self, x):
                return torch.histc(x, bins=self.bins)

            def forward_out(self, x, out):
                return torch.histc(x, bins=self.bins, min=self.min, max=self.max, out=out), out

        return aten_histc(bins, min, max, out, omit), "aten::histc"

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("bins", [1, 10, 100])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_basic(self, dtype, bins, ie_device, precision, ir_version):
        self._test(*self.create_model(bins, -10.0, 10.0), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": dtype, "shape": (50,), "low": -10.0, "high": 10.0, "bins": bins})

    @pytest.mark.parametrize("shape", [(50,), (7, 8)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_default_range(self, shape, ie_device, precision, ir_version):
        self._test(*self.create_model(10, 0.0, 0.0), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float32", "shape": shape, "low": -5.0, "high": 5.0, "bins": 10})

    @pytest.mark.parametrize("omit", ["all", "range"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_defaulted_args(self, omit, ie_device, precision, ir_version):
        # torch.export keeps defaulted args out of the FX node, so the translator must supply them.
        self._test(*self.create_model(8, 0.0, 0.0, omit=omit), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float32", "shape": (50,), "low": -5.0, "high": 5.0, "bins": 8})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_degenerate_literal_min_eq_max(self, ie_device, precision, ir_version):
        # ATen auto-ranges from data whenever min == max, even for an explicit nonzero pair.
        self._test(*self.create_model(4, 5.0, 5.0), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float32", "shape": (30,), "low": -3.0, "high": 3.0, "bins": 4})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_constant_input(self, ie_device, precision, ir_version):
        # all-equal input forces the degenerate +/-1 widening branch.
        self._test(*self.create_model(4, 0.0, 0.0), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float32", "fixed": [5.0, 5.0, 5.0], "bins": 4})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_empty_input(self, ie_device, precision, ir_version):
        self._test(*self.create_model(5, 0.0, 0.0), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float32", "fixed": [], "bins": 5})

    @pytest.mark.parametrize("out", [False, skip_if_export(True)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_out_variant(self, out, ie_device, precision, ir_version):
        self._test(*self.create_model(4, 0.0, 3.0, out=out), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float32", "shape": (10,), "low": 0.0, "high": 3.0,
                                             "bins": 4, "out": out})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_nan_and_out_of_range(self, ie_device, precision, ir_version):
        self._test(*self.create_model(4, 0.0, 3.0), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float32",
                                             "fixed": [-5.0, 1.0, float("nan"), 2.0, 10.0], "bins": 4})

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_dtype_coverage(self, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(8, -4.0, 4.0), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": dtype, "shape": (40,), "low": -4.0, "high": 4.0, "bins": 8})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_low_precision(self, ie_device, precision, ir_version):
        # Bin-centre values only: torch builds fp16 bin edges with an fp16 linspace plus a local search,
        # so elements near an edge are not reproducible by plain arithmetic.
        centres = [0.5, 1.5, 2.5, 3.5] * 5
        self._test(*self.create_model(4, 0.0, 4.0), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float16", "fixed": centres, "bins": 4})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_index_math_not_in_low_precision(self, ie_device, precision):
        # (x - min) * bins / span overflows fp16 to inf once span * bins > 65504, dropping
        # samples on any device that honours fp16 arithmetic. CPU silently up-converts, so the
        # promotion has to be asserted on the graph rather than on inference results.
        data = np.zeros((8,), dtype="float16")
        model, _ = self.create_model(100, -500.0, 500.0)
        om = ov.convert_model(torch.jit.script(model), example_input=(torch.from_numpy(data),))
        left_in_fp16 = [n.get_type_name() for n in om.get_ordered_ops()
                        if n.get_type_name() in ("Multiply", "Divide", "Floor")
                        and n.output(0).get_element_type() == ov.Type.f16]
        assert not left_in_fp16, f"bin index math left in fp16: {left_in_fp16}"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(PytorchLayerTest.use_torch_export(),
                        reason="data-dependent bins cannot be exported")
    def test_histc_dynamic_bins(self, ie_device, precision):
        # bins read from a tensor takes the non-constant OneHot-depth fallback.
        class aten_histc_dyn_bins(torch.nn.Module):
            def forward(self, x, n):
                return torch.histc(x, bins=int(n.item()), min=-4.0, max=4.0)

        data = self.random.uniform(-4.0, 4.0, size=(40,)).astype("float32")
        bins = np.array(8, dtype=np.int64)
        model = aten_histc_dyn_bins()
        ref = model(torch.from_numpy(data), torch.from_numpy(bins)).numpy()
        om = ov.convert_model(torch.jit.script(model),
                              example_input=(torch.from_numpy(data), torch.from_numpy(bins)))
        got = ov.Core().compile_model(om, ie_device)([data, bins])[0]
        assert np.array_equal(got, ref), f"ov={got} torch={ref}"

    @pytest.mark.parametrize("num_experts", [4, 8, 64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_integer_valued_data(self, num_experts, ie_device, precision, ir_version):
        # transformers' MoE routing shape: histc(expert_ids.float(), bins=E, min=0, max=E-1).
        ids = [float(i % num_experts) for i in range(3 * num_experts)]
        self._test(*self.create_model(num_experts, 0.0, float(num_experts - 1)), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float32", "fixed": ids, "bins": num_experts})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_integer_input(self, ie_device, precision):
        # ATen rejects integer input on CPU but supports it on CUDA, and transformers' MoE
        # routing feeds expert_ids.int() on the non-CPU path, so conversion must accept it.
        # No torch CPU reference exists, hence the bincount comparison.
        num_experts = 8
        ids = self.random.randint(0, num_experts, size=(200,)).astype(np.int32)

        class aten_histc_int(torch.nn.Module):
            def forward(self, x):
                return torch.histc(x, bins=8, min=0, max=7)

        model = aten_histc_int()
        om = ov.convert_model(torch.jit.script(model), example_input=(torch.from_numpy(ids),))
        got = ov.Core().compile_model(om, ie_device)([ids])[0]
        assert np.array_equal(got.astype(np.int64), np.bincount(ids, minlength=num_experts))

    @pytest.mark.parametrize("bins", [0, -5])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_error_bins_invalid(self, bins, ie_device, precision):
        # bins <= 0 is rejected eagerly by ATen itself, so script (not trace) to reach FE validation.
        model, _ = self.create_model(bins, 0.0, 3.0)
        sample_input = torch.from_numpy(self.random.uniform(0.0, 3.0, size=(10,)).astype(np.float32))
        with pytest.raises(ov.frontend.OpConversionFailure, match="histc"):
            ov.convert_model(torch.jit.script(model), example_input=(sample_input,))

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_error_min_greater_max(self, ie_device, precision):
        model, _ = self.create_model(4, 5.0, 2.0)
        sample_input = torch.from_numpy(self.random.uniform(0.0, 3.0, size=(10,)).astype(np.float32))
        with pytest.raises(ov.frontend.OpConversionFailure, match="histc"):
            ov.convert_model(torch.jit.script(model), example_input=(sample_input,))

    @pytest.mark.parametrize("bins", [3, 17, 64])
    @pytest.mark.parametrize("size", [1, 37, 199])
    @pytest.mark.nightly
    def test_histc_stress(self, bins, size, ie_device, precision, ir_version):
        self._test(*self.create_model(bins, -95.0, 95.0), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "float32", "shape": (size,), "low": -100.0, "high": 100.0,
                                             "bins": bins})
