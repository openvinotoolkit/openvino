# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import openvino as ov
import pytest
import torch

transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("Install transformers_tests/requirements.txt", allow_module_level=True)

from primitives import make_primitive, primitive_cases  # noqa: E402


class PrimitiveOutputs(torch.nn.Module):
    def __init__(self, primitive):
        super().__init__()
        self.primitive = primitive

    def forward(self, *inputs):
        outputs = self.primitive(*inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        return (*outputs, *inputs)


def primitive_parameters():
    parameters = []
    for case in primitive_cases():
        for mode in ("trace", "export"):
            reason = None
            if case.family == "cache":
                if mode == "export" and case.name.startswith("static-reorder"):
                    reason = "Static cache reorder has a data-dependent guard rejected by torch.export"
                elif mode == "trace" and case.name.startswith("static_sliding") and case.name.endswith("prefill"):
                    reason = "Static sliding cache tracing changes the returned prefill sequence length"
            elif case.family == "rope":
                if mode == "export" and case.name.startswith(("dynamic", "longrope")):
                    reason = "RoPE frequency updates have data-dependent guards rejected by torch.export"
                elif mode == "trace" and case.name == "dynamic-long":
                    reason = "Dynamic RoPE growth changes the graph between TorchScript trace invocations"
                elif mode == "trace" and case.name == "longrope-short":
                    reason = "TorchScript tracing rejects shared longrope frequency buffers"
            elif case.family == "generation":
                if mode == "trace" and case.name == "MinPLogitsWarper":
                    reason = "TorchScript alias analysis rejects boolean-scalar aten::scatter_"
                elif mode == "export" and case.name == "EtaLogitsWarper":
                    reason = "Eta sampling has a data-dependent distribution guard rejected by torch.export"
            marks = [pytest.mark.xfail(reason=reason, strict=True, raises=(AssertionError, RuntimeError))] if reason else []
            parameters.append(pytest.param(case, mode, id=f"{case.family}-{case.name}-{mode}", marks=marks))
    return parameters


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("device", os.environ.get("TEST_DEVICE", "CPU").split(";"))
@pytest.mark.parametrize("case,mode", primitive_parameters())
def test_primitive(case, mode, device):
    model, inputs = make_primitive(case)
    model = PrimitiveOutputs(model.eval()).eval()
    with torch.no_grad():
        try:
            converted = ov.convert_model(model, example_input=inputs, dynamo=mode == "export")
        except Exception as error:
            if "OpConversionFailure" in str(error) or type(error).__name__ == "OpConversionFailure":
                pytest.fail(f"Primitive conversion failed: {error}")
            raise
        if mode == "trace":
            converted.reshape({i: ov.PartialShape(list(value.shape)) for i, value in enumerate(inputs)})
        expected = tuple(value.detach().numpy().copy() for value in model(*inputs))
    config = {"INFERENCE_PRECISION_HINT": "f32"} if device == "CPU" else {}
    compiled = ov.Core().compile_model(converted, device, config)
    actual = compiled([value.numpy() for value in inputs])
    assert len(expected) == len(actual)
    for reference, result in zip(expected, actual.values()):
        assert reference.shape == result.shape
        assert reference.dtype == result.dtype
        np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-4)
