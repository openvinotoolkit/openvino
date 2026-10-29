# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

NAN = float("nan")
INF = float("inf")


class TestHistc(PytorchLayerTest):

    def _prepare_input(self, data, dtype):
        return (np.array(data, dtype=dtype),)

    def create_model(self, bins, min, max, use_kwargs=True):
        class HistcKwargs(torch.nn.Module):
            def __init__(self, bins, min, max):
                super().__init__()
                self.bins = bins
                self.min = min
                self.max = max

            def forward(self, x):
                return torch.histc(x, bins=self.bins, min=self.min, max=self.max)

        class HistcPositional(HistcKwargs):
            def forward(self, x):
                return torch.histc(x, self.bins, self.min, self.max)

        return (HistcKwargs if use_kwargs else HistcPositional)(bins, min, max), "aten::histc"

    def _run(self, data, dtype, bins, min, max, ie_device, precision, ir_version,
             trace_model=False, dynamic_shapes=True, use_kwargs=True):
        self._test(*self.create_model(bins, min, max, use_kwargs),
                   ie_device, precision, ir_version,
                   trace_model=trace_model, dynamic_shapes=dynamic_shapes,
                   kwargs_to_prepare_input={"data": data, "dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("trace_model", [False, True])
    @pytest.mark.parametrize("use_kwargs", [False, True])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("bins,min,max", [
        (4, 0, 4),          # integer Scalar bounds
        (4, 0.0, 4.0),      # float Scalar bounds
        (1, 0.0, 4.0),      # single bin
        (4, -2.0, 6.0),     # negative lower bound, range wider than data
        (3, 1.5, 3.5),      # range narrower than data: outliers are ignored
        (4, 0.0, 0.0),      # range taken from data
        (4, 2.0, 2.0),      # equal non-zero bounds also take range from data
        (100, 0.0, 0.0),    # default-like bins
        (1000, 0.0, 0.0),   # many bins
    ])
    def test_histc(self, dtype, bins, min, max, use_kwargs, trace_model,
                   ie_device, precision, ir_version):
        data = [1.0, 2.0, 1.0, 4.0, 5.5, -3.0, 2.5, 3.25, 0.0]
        self._run(data, dtype, bins, min, max, ie_device, precision, ir_version,
                  trace_model=trace_model, use_kwargs=use_kwargs)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("shape", [(), (1,), (17,), (4, 5), (2, 3, 4), (2, 1, 3, 2, 2)])
    @pytest.mark.parametrize("dynamic_shapes", [False, True])
    def test_histc_shapes(self, shape, dynamic_shapes, ie_device, precision, ir_version):
        data = np.random.RandomState(0).uniform(-3.0, 7.0, size=shape)
        self._run(data, np.float32, 5, -2.0, 6.0, ie_device, precision, ir_version,
                  dynamic_shapes=dynamic_shapes)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("bins,min,max", [(10, 0.0, 0.0), (7, -1.0, 1.0), (50, -5.0, 5.0)])
    def test_histc_random(self, seed, bins, min, max, ie_device, precision, ir_version):
        data = np.random.RandomState(seed).randn(3, 64).astype(np.float32)
        self._run(data, np.float32, bins, min, max, ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("data,bins,min,max", [
        ([0.0, 4.0, 4.0], 4, 0.0, 4.0),                 # max is inclusive in the last bin
        ([0.0, 1.0, 2.0, 3.0, 4.0], 4, 0.0, 4.0),       # values exactly on bin edges
        ([-1.0, -0.5, 0.0, 0.5, 1.0], 4, -1.0, 1.0),    # symmetric range
        ([3.0, 3.0, 3.0], 4, 0.0, 0.0),                 # constant data, degenerate range
        ([3.0, 3.0, 3.0], 5, 3.0, 3.0),                 # constant data, equal bounds
        ([0.0, 0.0, 0.0], 4, 0.0, 0.0),                 # all zeros
        ([-7.0], 3, 0.0, 0.0),                          # single negative element
        ([5.0, 6.0], 3, 0.0, 1.0),                      # all values outside the range
        ([1e-6, 2e-6, 3e-6], 3, 0.0, 4e-6),             # tiny range
        ([1e20, 2e20, -3e20], 4, 0.0, 0.0),             # huge values
        ([1.0, 2.0, 3.0], 3, -1e30, 1e30),              # huge range puts all in the middle bin
    ])
    def test_histc_edge_values(self, data, bins, min, max, ie_device, precision, ir_version):
        self._run(data, np.float32, bins, min, max, ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("data", [
        [NAN, 1.0, 2.0],
        [INF, 1.0, 2.0],
        [-INF, 1.0, 2.0, INF, NAN],
    ])
    def test_histc_non_finite_values_with_explicit_range(self, data, ie_device, precision, ir_version):
        # torch ignores NaN/Inf elements when the range is given explicitly
        self._run(data, np.float32, 3, 0.0, 3.0, ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("min,max", [(0.0, 0.0), (0.0, 1.0)])
    @pytest.mark.parametrize("dynamic_shapes", [True, False])
    def test_histc_empty_input(self, min, max, dynamic_shapes, ie_device, precision, ir_version):
        self._run(np.zeros((0,)), np.float32, 3, min, max, ie_device, precision, ir_version,
                  dynamic_shapes=dynamic_shapes)

    @pytest.mark.nightly
    def test_histc_float16(self, ie_device, precision, ir_version):
        data = [1.0, 2.0, 1.0, 4.0, 5.5, -3.0, 2.5, 3.25, 0.0]
        self._run(data, np.float16, 4, 0.0, 4.0, ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_defaults(self, ie_device, precision, ir_version):
        class HistcDefaults(torch.nn.Module):
            def forward(self, x):
                return torch.histc(x)

        data = np.random.RandomState(0).uniform(-1.0, 1.0, size=(200,))
        self._test(HistcDefaults(), "aten::histc", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"data": data, "dtype": np.float32})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_only_bins(self, ie_device, precision, ir_version):
        class HistcBins(torch.nn.Module):
            def forward(self, x):
                return torch.histc(x, bins=8)

        data = np.random.RandomState(1).uniform(-1.0, 1.0, size=(5, 40))
        self._test(HistcBins(), "aten::histc", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"data": data, "dtype": np.float32})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_in_graph(self, ie_device, precision, ir_version):
        # histc output feeds further ops, so output shape/type must be inferred correctly
        class HistcInGraph(torch.nn.Module):
            def forward(self, x):
                h = torch.histc(x * 2.0, bins=6, min=-4.0, max=4.0)
                return h / h.sum().clamp(min=1.0)

        data = np.random.RandomState(2).randn(4, 25)
        self._test(HistcInGraph(), "aten::histc", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"data": data, "dtype": np.float32})


# TODO: torch.histc has no integer CPU kernel (CUDA only), so integer inputs are not tested here.
# Add integer tests compared against torch once GPU support is added.
