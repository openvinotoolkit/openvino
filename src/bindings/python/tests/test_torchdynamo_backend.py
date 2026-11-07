# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Regression tests for the torch.compile "openvino" backend."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "openvino.frontend.pytorch.torchdynamo.backend",
    reason="PyTorch frontend not available",
)


@pytest.fixture
def _disable_suppress_errors():
    saved = torch._dynamo.config.suppress_errors
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.reset()
    yield
    torch._dynamo.config.suppress_errors = saved
    torch._dynamo.reset()


@pytest.mark.usefixtures("_disable_suppress_errors")
@pytest.mark.parametrize(
    "model_fn",
    [
        pytest.param(lambda t: t[0, ::2], id="step_slice"),
        pytest.param(lambda t: t[1:, :3], id="range_slice"),
        pytest.param(lambda t: torch.diagonal(t, offset=2), id="diag_offset"),
        pytest.param(lambda t: torch.diagonal(t, offset=-2), id="diag_neg_offset"),
        pytest.param(lambda t: torch.diagonal(t, dim1=-1, dim2=-2), id="diag_neg_dims"),
    ],
)
def test_compile_openvino_backend_does_not_break_dynamo(model_fn):
    x = torch.randn(4, 5, 6)
    expected = model_fn(x)
    actual = torch.compile(model_fn, backend="openvino")(x)
    torch.testing.assert_close(actual, expected)
