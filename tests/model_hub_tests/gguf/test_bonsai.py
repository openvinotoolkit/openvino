# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in complete Bonsai checkpoint catalog, including all published GGUF tensor mixes."""
import json
import os
from pathlib import Path

import numpy as np
import openvino as ov
import pytest

from huggingface_hub import hf_hub_download
from utils import assert_valid_logits, convert_gguf_model, _run_two_steps

CATALOG = json.loads(Path(__file__).with_name("bonsai_models.json").read_text())
MODELS = [m for m in CATALOG["models"] if m["kind"] != "mmproj"]


@pytest.mark.parametrize("checkpoint", MODELS, ids=[m["filename"] for m in MODELS])
def test_bonsai_checkpoint(checkpoint, ie_device):
    if os.environ.get("GGUF_BONSAI_ALL") != "1":
        pytest.skip("Set GGUF_BONSAI_ALL=1 on a runner sized for 27B F16 checkpoints")
    path = hf_hub_download(checkpoint["repo_id"], checkpoint["filename"], revision=checkpoint["revision"])
    model = convert_gguf_model(path)
    compiled = ov.Core().compile_model(
        model,
        ie_device,
        {
            "INFERENCE_PRECISION_HINT": "f32",
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
        },
    )
    if checkpoint["kind"] == "decoder":
        info = model.get_rt_info()
        pairs = info["gguf_recurrent_states"].value if "gguf_recurrent_states" in info else []
        states = dict(zip(pairs[1::2], pairs[::2]))
        imrope = bool(info["gguf_is_imrope"].value) if "gguf_is_imrope" in info else False
        assert_valid_logits(_run_two_steps(compiled, imrope, states))
        return
    request = compiled.create_infer_request()
    block = compiled.input("inp_tokens").partial_shape[-1].get_length()
    width = compiled.input("dspark_ctx_feat").partial_shape[-1].get_length()
    heads = next(p.partial_shape[2].get_length() for p in compiled.inputs if p.get_any_name().startswith("cache_k_"))
    for rows in (2, 3):
        count = rows + block
        feed = {}
        for port in compiled.inputs:
            name = port.get_any_name()
            if name in ("inp_tokens", "dspark_prev_tokens"):
                value = np.arange(1, block + 1, dtype=np.int32).reshape(1, 1, 1, block)
            elif name == "inp_pos":
                value = np.arange(count, dtype=np.int32).reshape(1, 1, 1, count)
            elif name == "inp_kv_idx":
                value = np.arange(count * heads, dtype=np.int32).reshape(1, 1, 1, count * heads)
            elif name == "dspark_ctx_feat":
                value = (0.1 * np.sin(np.arange(rows * width, dtype=np.float32))).reshape(1, 1, rows, width)
            else:
                shape = [d.get_length() if d.is_static else count for d in port.partial_shape]
                value = np.zeros(shape, dtype=port.element_type.to_dtype())
            feed[name] = value
        outputs = request.infer(feed)
        assert_valid_logits(outputs[compiled.output("logits")])
        for value in outputs.values():
            assert np.isfinite(value).all()
