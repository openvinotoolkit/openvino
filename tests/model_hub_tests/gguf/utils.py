# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv

import numpy as np
import openvino as ov
from openvino.frontend import FrontEndManager

from models_hub_common.constants import hf_cache_dir


def parse_gguf_model_list(file_name: str):
    """Parse a gguf_models_* list file into (arch, repo_id, filename, mark, reason) tuples.

    Each non-comment, non-empty line is "arch,repo_id,filename[,mark,reason]" (mark/reason
    may be present but empty). repo_id/filename are empty for a fully skipped entry that has
    no runnable checkpoint at all (see gguf_models_nightly).
    """
    models = []
    with open(file_name, "r") as f_in:
        for row in csv.reader(f_in):
            if not row or not row[0] or row[0].startswith("#"):
                continue
            row = [item.strip() for item in row]
            row += [""] * (5 - len(row))
            arch, repo_id, filename, mark, reason = row[:5]
            models.append((arch, repo_id, filename, mark or None, reason or None))
    return models


def gguf_hub_download(repo_id: str, filename: str) -> str:
    """Download one .gguf file from the HF hub into the shared HF cache dir."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=hf_cache_dir)


def _kv_cache_head_count(compiled_model) -> int:
    """Number of KV heads, read from any cache_k_* input port's static heads dim.

    All per-layer KV caches use the same [1, ctx, n_head_kv, head_size] layout (see
    src/frontends/gguf/src/builder/blocks/attention.cpp), so any one of them gives the count
    needed to size inp_kv_idx below. Falls back to 1 if the model has no KV cache at all
    (e.g. a purely recurrent / linear-attention stack).
    """
    for port in compiled_model.inputs:
        name = next(iter(port.get_names()), "")
        if name.startswith("cache_k_"):
            heads_dim = port.get_partial_shape()[2]
            if heads_dim.is_static:
                return heads_dim.get_length()
    return 1


def build_single_token_inputs(compiled_model, token_id: int = 1, is_imrope: bool = False):
    """Build a minimal, single-new-token input feed for the GGUF frontend's stateless IO
    contract (inp_tokens / inp_pos / inp_out_ids / self_kq_mask[_swa] / token_len_per_seq /
    inp_kv_idx), plus zero-filled tensors for every other input the model declares (per-layer
    KV caches and, for hybrid/recurrent architectures such as qwen35, the fully-static conv /
    delta state Parameters).

    This deliberately does not model real multi-token generation or attend to any real
    prompt/tokenizer: with T=1 (one new token, no prior context) every KV cache write covers
    the cache in full, so the frontend's default (caller-extension-free) SetRows lowering
    -- a plain ScatterUpdate over the whole cache -- is well-defined without requiring the
    caller to also register the MakeStateful extension. That keeps this check to exactly
    what "verify basic correctness" needs: the model converts, compiles, and produces a
    finite, correctly shaped output for one forward pass.

    is_imrope selects the interleaved M-RoPE position layout (qwen35 / qwen3vl), read from
    the model's "gguf_is_imrope" rt_info by the caller (CompiledModel does not expose
    rt_info, so this must be read from the ov.Model before compiling): such a model expects
    inp_pos to carry 4 position sections per token instead of 1.
    """
    n_head_kv = _kv_cache_head_count(compiled_model)
    n_pos_sections = 4 if is_imrope else 1
    feed = {}
    for port in compiled_model.inputs:
        name = next(iter(port.get_names()), None)
        assert name, "GGUF model input port has no name"

        if name == "inp_tokens":
            feed[name] = ov.Tensor(np.array([[[[token_id]]]], dtype=np.int32))
        elif name == "inp_pos":
            feed[name] = ov.Tensor(np.zeros((1, 1, 1, n_pos_sections), dtype=np.int32))
        elif name == "inp_out_ids":
            feed[name] = ov.Tensor(np.array([[[[0]]]], dtype=np.int32))
        elif name in ("self_kq_mask", "self_kq_mask_swa"):
            feed[name] = ov.Tensor(np.zeros((1, 1, 1, 1), dtype=np.float32))
        elif name == "token_len_per_seq":
            feed[name] = ov.Tensor(np.array([1], dtype=np.int64))
        elif name == "beam_idx":
            feed[name] = ov.Tensor(np.zeros((1,), dtype=np.int32))
        elif name == "inp_kv_idx":
            # One write index per (new token, kv head): translate_set_rows flattens the new
            # rows to [.., 1, tokens * n_head_kv, row_size] before the default ScatterUpdate
            # lowering, so indices must cover that same flattened extent.
            feed[name] = ov.Tensor(np.arange(n_head_kv, dtype=np.int32).reshape(1, 1, 1, n_head_kv))
        else:
            # Any other input (a per-layer KV cache, or a recurrent/linear-attention state
            # such as qwen35's Gated-DeltaNet conv window and delta matrix): zero-fill,
            # resolving the one dynamic (token/context) axis, if any, to T=1.
            partial_shape = port.get_partial_shape()
            shape = [d.get_length() if d.is_static else 1 for d in partial_shape]
            dtype = port.get_element_type().to_dtype()
            feed[name] = ov.Tensor(np.zeros(shape, dtype=dtype))
    return feed


def convert_gguf_model(path: str) -> ov.Model:
    """Convert a .gguf into an ov.Model through the GGUF frontend.

    The frontend is deliberately not auto-selectable (see is_hidden_frontend in
    src/frontends/common/src/manager.cpp), so core.read_model(".gguf") does not resolve to it and
    the frontend has to be asked for by name. Nothing else is needed: the frontend runs its own
    normalization inside convert(), and the only step read_model would add on top,
    update_v10_model, applies solely to legacy IR v10 models.
    """
    fe = FrontEndManager().load_by_framework("gguf")
    return fe.convert(fe.load(path))


def run_gguf_model(repo_id: str, filename: str, device: str = "CPU"):
    """Download, convert, compile, and run one forward step of a .gguf model.

    Returns the raw logits tensor as a numpy array. Raises on any conversion / compilation /
    inference failure; the caller is expected to additionally sanity-check the result (shape,
    finiteness) -- see assert_valid_logits below.
    """
    path = gguf_hub_download(repo_id, filename)
    core = ov.Core()
    model = convert_gguf_model(path)
    is_imrope = False
    if "gguf_is_imrope" in model.get_rt_info():
        is_imrope = bool(model.get_rt_info(["gguf_is_imrope"]).get())
    compiled_model = core.compile_model(model, device)
    request = compiled_model.create_infer_request()
    feed = build_single_token_inputs(compiled_model, is_imrope=is_imrope)
    outputs = request.infer(feed)
    return next(iter(outputs.values()))


def assert_valid_logits(logits: np.ndarray):
    """Basic correctness: a real (non-empty), finite tensor of logits over some vocabulary.

    Deliberately does not check the actual predicted token / text: this suite verifies the
    frontend converts and runs each supported architecture, not that any given (quantized,
    single-token, no-history) forward pass produces a meaningful continuation.
    """
    assert logits.size > 0, "model produced an empty output tensor"
    assert np.isfinite(logits).all(), "model output contains NaN/Inf"
    vocab_size = logits.shape[-1]
    assert vocab_size > 1, f"output's last dim ({vocab_size}) does not look like a vocab axis"
    # argmax must be a valid vocabulary index -- mostly a sanity check on the shape/dtype
    # plumbing above, since any finite tensor trivially satisfies this.
    best = int(logits.reshape(-1, vocab_size)[-1].argmax())
    assert 0 <= best < vocab_size
