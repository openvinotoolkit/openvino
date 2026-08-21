# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Capture of PyTorch models whose output or control flow needs torch.export."""

import typing
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from openvino import Core, convert_model


@dataclass
class PlainDataclassOutput:
    first: torch.Tensor
    second: torch.Tensor
    empty: typing.Optional[torch.Tensor] = None


class DataclassOutputModel(torch.nn.Module):
    def forward(self, x):
        return PlainDataclassOutput(first=x + 1, second=x * 2)


class VmapMaskModel(torch.nn.Module):
    """Mask construction pattern used by recent `transformers` releases.

    ``vmap`` over ``TransformGetItemToIndex`` dispatches to a custom
    ``autograd.Function``. The TorchScript tracer cannot record it, while
    ``torch.export`` captures it and keeps the sequence length dynamic.
    """

    def forward(self, x, pad):
        try:
            from torch._higher_order_ops.flex_attention import TransformGetItemToIndex
        except ImportError:
            from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

        idx = torch.arange(x.shape[-1])
        with TransformGetItemToIndex():
            mask = torch.vmap(lambda i: pad[i])(idx)
        return x * mask.to(x.dtype)


@pytest.mark.parametrize("dynamo", [False, True])
def test_dataclass_output(dynamo):
    """A model returning a plain ``@dataclass`` converts on both capture paths."""
    ov_model = convert_model(DataclassOutputModel(),
                             example_input=torch.randn(1, 4), dynamo=dynamo)
    # `empty` is None and must not become an output
    assert len(ov_model.outputs) == 2


def test_untraceable_model_suggests_dynamo():
    """A tracing failure points at the capture path that can handle the model."""
    model = VmapMaskModel().eval()
    x = torch.randn(1, 5)
    pad = torch.ones(5, dtype=torch.int64)

    with pytest.raises(Exception, match="dynamo=True"):
        convert_model(model, example_input=(x, pad))

    ov_model = convert_model(model, example_input=(x, pad), dynamo=True)
    compiled = Core().compile_model(ov_model, "CPU")
    assert np.allclose(compiled((x.numpy(), pad.numpy()))[0],
                       model(x, pad).detach().numpy())
