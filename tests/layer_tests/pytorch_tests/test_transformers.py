# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("Install requirements_transformers for Transformers primitive coverage", allow_module_level=True)

from transformers_primitives import make_primitive, primitive_cases  # noqa: E402


class PrimitiveOutputs(torch.nn.Module):
    def __init__(self, primitive):
        super().__init__()
        self.primitive = primitive

    def forward(self, *inputs):
        outputs = self.primitive(*inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        # Preserve inputs that torch.export would otherwise remove.
        return (*outputs, *inputs)


def expected_kind(case):
    return {
        "activation": {"gelu": "aten::gelu", "gelu_10": "aten::clip", "gelu_fast": "aten::tanh",
            "gelu_new": "aten::tanh", "gelu_python": "aten::erf", "gelu_pytorch_tanh": "aten::gelu",
            "gelu_python_tanh": "aten::tanh", "gelu_accurate": "aten::tanh", "hardswish": "aten::hardswish",
            "laplace": "aten::erf", "leaky_relu": "aten::leaky_relu", "linear": None,
            "mish": "aten::mish", "quick_gelu": "aten::sigmoid", "relu": "aten::relu",
            "relu2": "aten::relu", "relu6": "aten::hardtanh", "sigmoid": "aten::sigmoid",
            "silu": "aten::silu", "sqrtsoftplus": "aten::sqrt", "swish": "aten::silu",
            "tanh": "aten::tanh", "prelu": "aten::prelu", "xielu": "aten::where"},
        "utility": {"conv1d": "aten::addmm", "chunking": "aten::cat", "meshgrid": "aten::meshgrid",
                    "rms_norm": "aten::rsqrt", "gated_mlp": "aten::linear", "relative_position": "aten::where"},
        "mask": "aten::where" if case.name.startswith("eager") else "aten::__and__",
        "attention": "aten::scaled_dot_product_attention" if case.name.startswith("sdpa") else "aten::matmul",
        "cache": "aten::cat" if case.name.startswith(("dynamic", "sliding")) else "aten::roll"
            if case.name.startswith("static_sliding") and case.name.endswith("decode") else "aten::index_copy_",
        "rope": "aten::neg", "vision": {"patch_embedding": "aten::_convolution",
            "patch_interpolation": "aten::upsample_bicubic2d", "sine_position": "aten::cumsum",
            "window_partition": "aten::transpose"},
        "audio": "aten::_convolution", "time_series": {"patchify": "aten::unfold", "std_scaler": "aten::sqrt",
                                                         "mean_scaler": "aten::abs"},
        "generation": {"TemperatureLogitsWarper": "aten::div", "TopKLogitsWarper": "aten::topk",
            "TopPLogitsWarper": "aten::sort", "MinPLogitsWarper": "aten::softmax",
            "TypicalLogitsWarper": "aten::sort", "EpsilonLogitsWarper": "aten::softmax",
            "EtaLogitsWarper": "aten::softmax", "RepetitionPenaltyLogitsProcessor": "aten::gather",
            "MinLengthLogitsProcessor": "aten::where", "ForcedBOSTokenLogitsProcessor": "aten::full_like",
            "ForcedEOSTokenLogitsProcessor": "aten::full_like", "SuppressTokensLogitsProcessor": "aten::where",
            "InfNanRemoveLogitsProcessor": "aten::where", "LogitNormalization": "aten::log_softmax"},
    }[case.family]


def known_failure(case):
    exporting = PytorchLayerTest.use_torch_export()
    name, family = case.name, case.family
    if family == "activation" and name in {"sqrtsoftplus", "xielu"}:
        return "Activation accuracy differs from PyTorch at rtol=atol=1e-4"
    if family == "attention" and exporting and name.startswith("sdpa-2-") and name.endswith("False"):
        return "Export SDPA conversion does not repeat grouped key/value heads"
    if family == "cache":
        if name.startswith(("dynamic", "sliding")):
            return "Cache conversion cannot concatenate the initial rank-one empty cache with rank-four states"
        if exporting and name.startswith("static-reorder"):
            return "Static cache reorder hits torch.export GuardOnDataDependentSymNode"
        if exporting and name == "static_sliding-update-decode":
            return "Export static sliding cache output triggers Tensor name is not a number: copy"
        if exporting and name.startswith("static-update"):
            return "Export cache outputs trigger Tensor name is not a number in the PyTorch frontend"
    if family == "rope":
        if exporting and name.startswith(("dynamic", "longrope")):
            return "Dynamic RoPE frequency updates hit torch.export GuardOnDataDependentSymNode"
        if not exporting and name == "longrope-short":
            return "TorchScript tracing rejects shared longrope frequency buffers"
    if family == "time_series" and name == "patchify":
        return "Time-series aten::unfold conversion produces an invalid transpose order"
    if family == "generation":
        if name == "ForcedEOSTokenLogitsProcessor":
            return "Forced EOS scalar aten::index_put_ conversion produces an invalid transpose order"
        if not exporting and name in {"MinLengthLogitsProcessor", "SuppressTokensLogitsProcessor"}:
            return "OpenVINO does not convert aten::isin used by token suppression"
        if not exporting and name == "TypicalLogitsWarper":
            return "OpenVINO does not convert aten::nansum used by typical sampling"
        if not exporting and name == "MinPLogitsWarper":
            return "TorchScript alias analysis rejects aten::scatter_ with a boolean scalar"
        if exporting and name == "EtaLogitsWarper":
            return "Eta sampling hits torch.export GuardOnDataDependentSymNode"
    return None


def primitive_parameters():
    cases = []
    for case in primitive_cases():
        reason = known_failure(case)
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        cases.append(pytest.param(case, id=f"{case.family}-{case.name}", marks=marks))
    return cases


class TestTransformersPrimitives(PytorchLayerTest):
    def _prepare_input(self):
        return tuple(value.numpy() for value in self.example)

    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.nightly
    @pytest.mark.parametrize("case", primitive_parameters())
    def test_transformers_primitive(self, case, ie_device, precision, ir_version):
        model, self.example = make_primitive(case)
        kind = expected_kind(case)
        if isinstance(kind, dict):
            kind = kind[case.name]
        fx_kind = "aten.conv1d" if case.family == "audio" else None
        if case.family == "vision" and case.name == "patch_embedding":
            fx_kind = "aten.conv2d"
        if kind is None:
            fx_kind = []
        self._test(PrimitiveOutputs(model.eval()), kind, ie_device, precision, ir_version,
                   trace_model=True, dynamic_shapes=False, freeze_model=False, fx_kind=fx_kind)
