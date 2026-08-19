# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the standalone GGUF frontend scripts (compare_with_llama.py, bench_gguf.py):
# converting a .gguf through the frontend and driving its stateless gguf-IO contract
# (inp_tokens / inp_pos / self_kq_mask / token_len_per_seq + a per-layer KV cache) from a plain
# token sequence, the way a genai-side wrapper would.

import numpy as np
import openvino as ov
from openvino.frontend import FrontEndManager


def convert_gguf(model_path: str):
    """Convert a .gguf through the GGUF frontend.

    The frontend is not auto-selectable (see is_hidden_frontend in
    src/frontends/common/src/manager.cpp), so core.read_model(".gguf") does not reach it and it
    has to be requested by name.

    No DecoderTransformationExtension is registered (GGUFMakeStateful is a C++-only pass with
    no Python binding), so this returns the frontend's plain STATELESS graph: every per-layer KV
    cache is a Parameter/Result pair rather than an OpenVINO Variable, and there is no beam_idx.
    KVCache below round-trips those Parameters/Results across decode steps in Python instead.
    """
    fe = FrontEndManager().load_by_framework("gguf")
    return fe.convert(fe.load(model_path))


class KVCache:
    """Round-trips the stateless graph's per-layer KV cache Parameter/Result pairs across decode
    steps, since no DecoderTransformationExtension is registered to fold them into OpenVINO state
    (see convert_gguf).

    The frontend lowers each layer's SET_ROWS(cur, inp_kv_idx, cache) to a ScatterUpdate(cache,
    inp_kv_idx, cur) (see lower_set_rows_stateless.cpp): the cache Parameter fed to infer() must
    already be sized to cover every index ScatterUpdate writes this step, with the new rows'
    content otherwise irrelevant (they are unconditionally overwritten). So each step this grows
    every buffer by the new-token count with placeholder zero rows, feeds inp_kv_idx = the
    absolute positions of those new rows (matching inp_pos), and keeps the Result -- the same
    buffer with those rows now holding real K/V -- as the next step's input. This mirrors what
    GGUFMakeStateful's ReadValue/Concat/Assign would do inside the graph.

    Only "own-KV" layers (see attention.cpp's has_own_kv / shared_kv_layers) get a
    cache_k_l{il}/cache_v_l{il} Parameter+Result pair; a shared-KV layer (gemma4) has none of its
    own, so discovering the buffers from the compiled model's input names naturally covers every
    distinct cache without needing the layer/shared-KV bookkeeping DecoderConfig has internally.
    """

    def __init__(self, compiled):
        self._names = sorted(n for p in compiled.inputs for n in p.get_names() if n.startswith("cache_"))
        self._dtypes = {n: compiled.input(n).get_element_type().to_dtype() for n in self._names}
        # Static dims of the rank-4 cache Parameter [batch=1, dynamic token axis, n_head_kv,
        # head_size] (see attention.cpp's cache_shape).
        self._shapes = {}
        for n in self._names:
            ps = compiled.input(n).get_partial_shape()
            self._shapes[n] = (ps[2].get_length(), ps[3].get_length())
        self._buf = {n: np.zeros((1, 0, *self._shapes[n]), dtype=self._dtypes[n]) for n in self._names}

    def inputs_for_step(self, n_new):
        feed = {}
        for name in self._names:
            new_rows = np.zeros((1, n_new, *self._shapes[name]), dtype=self._dtypes[name])
            self._buf[name] = np.concatenate([self._buf[name], new_rows], axis=1)
            feed[name] = ov.Tensor(self._buf[name])
        return feed

    def update_from(self, req):
        for name in self._names:
            self._buf[name] = np.array(req.get_tensor(name).data)


def build_inputs(tokens, past_len):
    """Build the gguf-IO tensors for one decode step, as plain numpy arrays.

    tokens   : list[int] of the new token ids for this step.
    past_len : number of tokens already in the KV cache.
    Returns a dict of input name -> np.ndarray; wrap in ov.Tensor at the call site (only the
    inputs the -- possibly pruned -- compiled model actually exposes need wrapping).
    """
    n = len(tokens)
    total = past_len + n
    inp_tokens = np.array(tokens, dtype=np.int32).reshape(1, 1, 1, n)
    inp_pos = np.arange(past_len, past_len + n, dtype=np.int32).reshape(1, 1, 1, n)
    # last-token logits only (matches llama-simple's per-step argmax on the final token)
    inp_out_ids = np.array([n - 1], dtype=np.int32).reshape(1, 1, 1, 1)
    # causal mask [1, 1, n, total]: 0 where attended, -inf where masked.
    mask = np.zeros((1, 1, n, total), dtype=np.float32)
    for i in range(n):
        # query token i (absolute position past_len + i) may attend to keys 0..past_len+i
        allowed = past_len + i + 1
        mask[0, 0, i, allowed:] = -np.inf
    token_len = np.array([n], dtype=np.int64)
    # beam_idx: identity beam reorder for the (single-beam, batch-1) stateful KV cache. Only
    # meaningful if a stateful model was produced (see convert_gguf); harmless otherwise, since
    # the caller drops any key absent from the model's actual inputs.
    beam_idx = np.zeros((1,), dtype=np.int32)
    # KV-cache write index for the stateless graph: the new rows' absolute positions, same
    # values as inp_pos (see KVCache).
    inp_kv_idx = np.arange(past_len, past_len + n, dtype=np.int32).reshape(1, 1, 1, n)
    return {
        "inp_tokens": inp_tokens,
        "inp_pos": inp_pos,
        "inp_out_ids": inp_out_ids,
        "self_kq_mask": mask,
        # gpt-oss sliding-window mask: for prompts shorter than the window it equals the
        # full causal mask, so the same tensor is correct here.
        "self_kq_mask_swa": mask.copy(),
        "token_len_per_seq": token_len,
        "beam_idx": beam_idx,
        "inp_kv_idx": inp_kv_idx,
    }
