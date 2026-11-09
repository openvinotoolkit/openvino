# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""load_format="nncf_hooks" turns NNCF's hook layout into the nncf layer parameters, for int4 and int8, sym and asym."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from openvino.frontend.pytorch.torchdynamo.vllm import model_loader
from openvino.frontend.pytorch.torchdynamo.vllm.quantization import _dequantize

OUT, IN, GROUP = 8, 16, 8
LINEAR, DENSE = "model.layers.0.mlp.down_proj", "lm_head"


def _pack(nibbles):
    """Two uint8 nibbles per byte along the last dim, low nibble first."""
    return nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)


def _nncf_tensors(name, bits, sym):
    """`name`.weight in NNCF's layout, and the dense weight it decompresses to."""
    groups = IN // GROUP if bits == 4 else 1
    levels = 1 << bits
    levels_q = torch.randint(0, levels, (OUT, IN), dtype=torch.uint8)
    scale = (torch.rand(OUT, groups, 1) * 0.1 + 0.01).to(torch.float16)
    zp = None if sym else torch.randint(0, levels, (OUT, groups, 1), dtype=torch.uint8)
    offset = levels // 2 if sym else zp.repeat_interleave(IN // groups, dim=1).reshape(OUT, IN).to(torch.float16)
    centered = (levels_q.to(torch.float16) - offset).reshape(OUT, groups, -1)
    dense = (centered * scale).reshape(OUT, IN).float()
    if bits == 8:
        stored = (levels_q.to(torch.int16) - offset).to(torch.int8) if sym else levels_q
        scale = scale.reshape(OUT, 1)
        zp = None if sym else zp.reshape(OUT, 1)
    else:
        stored = _pack(levels_q)
        zp = None if sym else _pack(zp.reshape(OUT, groups)).reshape(OUT, groups // 2, 1)
    tensors = {f"{name}.weight": stored, model_loader._hook_key(f"{name}.weight", "_scale"): scale}
    if zp is not None:
        tensors[model_loader._hook_key(f"{name}.weight", "_zero_point")] = zp
    return tensors, dense


@pytest.mark.precommit
@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("sym", [True, False], ids=["sym", "asym"])
def test_nncf_hooks_weights(tmp_path, bits, sym):
    torch.manual_seed(0)
    linear, linear_dense = _nncf_tensors(LINEAR, bits, sym)
    lm_head, lm_head_dense = _nncf_tensors(DENSE, bits, sym)
    save_file({**linear, **lm_head}, str(tmp_path / "model.safetensors"))
    # Only LINEAR is an nncf layer; lm_head (not a LinearBase) loads dense.
    params = {f"{LINEAR}.{p}": None for p in ("qweight", "scale", "zero_point")}
    model = SimpleNamespace(named_parameters=params.items, named_modules=lambda: [])
    model_config = SimpleNamespace(dtype=torch.float32, model=str(tmp_path),
                                   hf_config=SimpleNamespace(quantization_config={"bits": bits, "sym": sym}))

    loader = model_loader.register()
    weights = dict(loader._weights(object.__new__(loader), model, model_config))

    expected = {f"{LINEAR}.qweight", f"{LINEAR}.scale", f"{DENSE}.weight"}
    assert set(weights) == (expected if sym else expected | {f"{LINEAR}.zero_point"})
    torch.testing.assert_close(weights[DENSE + ".weight"], lm_head_dense)
    got = _dequantize(weights[f"{LINEAR}.qweight"], weights[f"{LINEAR}.scale"], bits, torch.float32,
                      zero_point=weights.get(f"{LINEAR}.zero_point"))
    torch.testing.assert_close(got, linear_dense)
