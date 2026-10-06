# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Generate small GGUF models and logits from real llama.cpp CPU, never from OpenVINO.

Usage: PYTHONPATH=<llama.cpp>/gguf-py python gen_arch_accuracy.py --oracle <architecture_oracle>
Build architecture_oracle.cpp against a CPU-only llama.cpp (include/, ggml/include/,
link libllama, libggml, libggml-base). The generated NPZs are consumed without llama.cpp.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

import gguf
import numpy as np

from mmproj_fixtures import finish, save_npz

CASES = {
    "qwen35": {}, "qwen35moe": {},
    "qwen35moe-fused": {"architecture": "qwen35moe", "fused_experts": True},
    # Real checkpoints quantize attn_gate apart from ssm_beta/ssm_alpha: F16 here selects the
    # frontend's beta+alpha-only merge, and an F16 beta in layer 2 selects separate projections.
    "qwen35-mixed": {"architecture": "qwen35", "f16": ("attn_gate", "blk.2.ssm_beta")},
    "nemotron_h": {},
    "mamba2": {}, "mamba2-tied": {"architecture": "mamba2", "tied": True},
    "llama": {},
    "llama-embed": {"embedding_output": True, "tied": True},
    "llama-embed-noncausal": {"architecture": "llama-embed", "embedding_output": True, "tied": True, "causal": False},
    "llama-embed-mean": {"architecture": "llama-embed", "embedding_output": True, "tied": True, "pooling": 1},
    "llama-embed-cls": {"architecture": "llama-embed", "embedding_output": True, "tied": True, "pooling": 2},
    "llama-embed-last": {"architecture": "llama-embed", "embedding_output": True, "tied": True, "pooling": 3},
    "exaone-moe": {"qk": True, "moe": True, "shared": True, "lead": 1, "swa": True, "layers": 4},
    "exaone-moe-nextn": {"architecture": "exaone-moe", "qk": True, "moe": True, "shared": True, "lead": 1, "swa": True, "layers": 4, "nextn": 1, "sigmoid": True, "selection_bias": True, "expert_scale": 2.5},
    "glm4moe": {"qk": True, "moe": True, "shared": True, "lead": 1, "selection_bias": True, "sigmoid": True, "ffn_post_attn": True},
    "jais2": {"layer_norm": True, "relu_squared": True, "bias": True},
    "minimax-m2": {"moe": True, "full_qk": True, "selection_bias": True, "rope_dims": 4, "sigmoid": True},
    "plamo3": {"qk": True, "fused": True, "fused_ffn": True, "post": True, "bare_post": True, "swa": True}, "qwen2": {"bias": True}, "qwen3": {"qk": True},
    "phi3": {"fused": True, "fused_ffn": True}, "minicpm": {},
    "olmoe": {"moe": True, "full_qk": True},
    "hunyuan-dense": {"qk": True}, "hunyuan-moe": {"qk": True, "moe": True, "shared": True},
    "qwen3moe": {"qk": True, "moe": True},
    "gemma4-mqa": {"architecture": "gemma4"},
    "gemma4-moe": {"architecture": "gemma4", "moe": True},
    "gemma4-ple": {"architecture": "gemma4", "per_layer": 8, "layers": 4, "shared_kv": 2},
    "gemma": {"mqa": True, "tied": True},
    "gemma2": {"post": True, "swa": True, "tied": True, "softcap": True},
    "gemma3": {"post": True, "swa": True, "tied": True, "qk": True, "linear": 8.0},
    "exaone4": {"qk": True, "post_only": True},
    "ernie4_5-moe": {"moe": True, "lead": 1, "selection_bias": True, "shared": True},
    "bailingmoe2": {"qk": True, "fused": True, "moe": True, "lead": 1,
                    "selection_bias": True, "shared": True, "sigmoid": True},
    "maincoder": {"qk": True}, "mistral3": {}, "smollm3": {"layers": 4},
    "mellum": {"qk": True, "moe": True},
    "muse-glimmer": {"qk": True, "post": True, "swa": True, "gate": True},
    "deepseek2-ocr": {"moe": True, "lead": 1, "shared": True},
    "devstral-small": {"architecture": "llama", "embedding": 40},
    "devstral-small2": {"architecture": "mistral3", "embedding": 40, "yarn": 48.0,
                       "yarn_log_mul": 1.0, "temperature": 0.1, "original_context": 2},
    "devstral2": {"architecture": "mistral3", "yarn": 64.0, "yarn_beta_fast": 4.0,
                 "yarn_log_mul": 0.0, "original_context": 128},
}


def tensor_writer(writer, rng, f16=()):
    """Uniform random F32 tensors; norm weights are centered at 1. Names containing an `f16`
    entry are stored as F16."""
    def tensor(name, shape, norm=False):
        values = rng.uniform(-1, 1, shape).astype(np.float32)
        values = 1 + values * .3 if norm else values * .2
        writer.add_tensor(name, values.astype(np.float16) if any(s in name for s in f16) else values)
    return tensor


def write_mamba2_model(path, opts, arch="mamba2"):
    w = gguf.GGUFWriter(path, arch)
    hybrid = arch == "nemotron_h"
    d, inner, heads, groups, state, kernel, vocab = 32, 64, 4, 2, 8, 4, 32
    w.add_context_length(128)
    w.add_embedding_length(d)
    w.add_block_count(4 if hybrid else 2)
    w.add_head_count(4 if hybrid else 0)
    if hybrid:
        w.add_head_count_kv([0, 2, 0, 0])
        w.add_feed_forward_length([0, 0, 48, 0])
        w.add_key_length(8)
        w.add_value_length(8)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(vocab)
    w.add_tokenizer_model("none")
    for key, value in {"inner_size": inner, "time_step_rank": heads, "group_count": groups,
                       "state_size": state, "conv_kernel": kernel}.items():
        w.add_uint32(arch + ".ssm." + key, value)
    rng = np.random.default_rng(20260915)
    tensor = tensor_writer(w, rng)

    tensor("token_embd.weight", (vocab, d))
    tensor("output_norm.weight", (d,), True)
    if opts.get("layer_norm"):
        tensor("output_norm.bias", (d,))
    if not opts.get("tied"):
        tensor("output.weight", (vocab, d))
    conv_dim = inner + 2 * groups * state
    for layer in range(4 if hybrid else 2):
        p = f"blk.{layer}."
        tensor(p + "attn_norm.weight", (d,), True)
        if hybrid and layer == 1:
            for name, width in (("q", 32), ("k", 16), ("v", 16)):
                tensor(p + f"attn_{name}.weight", (width, d))
                tensor(p + f"attn_{name}.bias", (width,))
            tensor(p + "attn_output.weight", (d, 32))
            tensor(p + "attn_output.bias", (d,))
            continue
        if hybrid and layer == 2:
            tensor(p + "ffn_up.weight", (48, d))
            tensor(p + "ffn_up.bias", (48,))
            tensor(p + "ffn_down.weight", (d, 48))
            tensor(p + "ffn_down.bias", (d,))
            continue
        tensor(p + "ssm_in.weight", (inner + conv_dim + heads, d))
        tensor(p + "ssm_conv1d.weight", (conv_dim, kernel))
        tensor(p + "ssm_conv1d.bias", (conv_dim,))
        tensor(p + "ssm_dt.bias", (heads,))
        w.add_tensor(p + "ssm_a", -rng.uniform(.1, 1, (heads, 1)).astype(np.float32))
        tensor(p + "ssm_d", (heads, 1))
        tensor(p + "ssm_norm.weight", (groups, inner // groups), True)
        tensor(p + "ssm_out.weight", (d, inner))
    finish(w)


def write_qwen35_model(path, arch, fused_experts=False, f16=()):
    w = gguf.GGUFWriter(path, arch)
    d, head, heads, kv, ff, vocab = 32, 16, 4, 2, 48, 32
    state, groups, vheads, kernel = 8, 2, 4, 4
    inner, conv = state * vheads, state * (2 * groups + vheads)
    w.add_context_length(128)
    w.add_embedding_length(d)
    w.add_block_count(4)
    w.add_feed_forward_length(ff)
    w.add_head_count(heads)
    w.add_head_count_kv(kv)
    w.add_key_length(head)
    w.add_value_length(head)
    w.add_rope_dimension_count(8)
    w.add_rope_freq_base(10000.)
    w.add_array(arch + ".rope.dimension_sections", [1, 1, 2, 0])
    w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(vocab)
    w.add_tokenizer_model("none")
    for key, value in {"inner_size": inner, "time_step_rank": vheads, "group_count": groups,
                       "state_size": state, "conv_kernel": kernel}.items():
        w.add_uint32(arch + ".ssm." + key, value)
    w.add_uint32(arch + ".full_attention_interval", 4)
    moe = arch == "qwen35moe"
    if moe:
        w.add_expert_count(4)
        w.add_expert_used_count(2)
        w.add_expert_feed_forward_length(ff)
        w.add_expert_shared_feed_forward_length(ff)
    rng = np.random.default_rng(20260922)
    tensor = tensor_writer(w, rng, f16)

    tensor("token_embd.weight", (vocab, d))
    tensor("output_norm.weight", (d,), True)
    tensor("output.weight", (vocab, d))
    for layer in range(4):
        p = f"blk.{layer}."
        tensor(p + "attn_norm.weight", (d,), True)
        tensor(p + "post_attention_norm.weight", (d,), True)
        if layer < 3:
            tensor(p + "attn_qkv.weight", (conv, d))
            tensor(p + "attn_gate.weight", (inner, d))
            tensor(p + "ssm_conv1d.weight", (conv, kernel))
            tensor(p + "ssm_dt.bias", (vheads,))
            w.add_tensor(p + "ssm_a", -rng.uniform(.5, 1.5, vheads).astype(np.float32))
            for name in ("beta", "alpha"):
                tensor(p + f"ssm_{name}.weight", (vheads, d))
            tensor(p + "ssm_norm.weight", (state,), True)
            tensor(p + "ssm_out.weight", (d, inner))
        else:
            for name, width in (("q", 2 * heads * head), ("k", kv * head), ("v", kv * head)):
                tensor(p + f"attn_{name}.weight", (width, d))
            for name in ("q", "k"):
                tensor(p + f"attn_{name}_norm.weight", (head,), True)
            tensor(p + "attn_output.weight", (d, heads * head))
        if moe:
            tensor(p + "ffn_gate_inp.weight", (4, d))
            if fused_experts:
                tensor(p + "ffn_gate_up_exps.weight", (4, 2 * ff, d))
            else:
                for name in ("gate", "up"):
                    tensor(p + f"ffn_{name}_exps.weight", (4, ff, d))
            tensor(p + "ffn_down_exps.weight", (4, d, ff))
            tensor(p + "ffn_gate_inp_shexp.weight", (d,))
        for name, shape in (("gate", (ff, d)), ("up", (ff, d)), ("down", (d, ff))):
            tensor(p + f"ffn_{name}" + ("_shexp" if moe else "") + ".weight", shape)
    finish(w)


def write_gemma4_model(path, opts):
    """Layers alternate sliding/full attention; E2B/E4B-style options add per-layer
    embeddings (`per_layer`) and reuse earlier KV in the last `shared_kv` layers, which keep
    their unused K/V tensors as real checkpoints do."""
    w = gguf.GGUFWriter(path, "gemma4")
    d, heads, layers, vocab, ff = 32, 4, opts.get("layers", 2), 32, 48
    per_layer, shared = opts.get("per_layer", 0), opts.get("shared_kv", 0)
    w.add_context_length(128)
    w.add_embedding_length(d)
    w.add_block_count(layers)
    w.add_head_count(heads)
    w.add_head_count_kv([2, 1] * (layers // 2))
    w.add_feed_forward_length(ff)
    w.add_key_length(16)
    w.add_value_length(16)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_vocab_size(vocab)
    w.add_tokenizer_model("none")
    for key, value in {"attention.key_length_swa": 8, "attention.value_length_swa": 8,
                       "attention.sliding_window": 2, "attention.shared_kv_layers": shared,
                       "embedding_length_per_layer_input": per_layer, "rope.dimension_count": 16,
                       "rope.dimension_count_swa": 8}.items():
        w.add_uint32("gemma4." + key, value)
    w.add_array("gemma4.attention.sliding_window_pattern", [True, False] * (layers // 2))
    w.add_float32("gemma4.rope.freq_base", 1000000.)
    w.add_float32("gemma4.rope.freq_base_swa", 10000.)
    w.add_float32("gemma4.final_logit_softcapping", 30.)
    if opts.get("moe"):
        w.add_expert_count(4)
        w.add_expert_used_count(2)
        w.add_uint32("gemma4.expert_feed_forward_length", 24)
    rng = np.random.default_rng(20260922)
    tensor = tensor_writer(w, rng)

    tensor("token_embd.weight", (vocab, d))
    tensor("output_norm.weight", (d,), True)
    tensor("rope_freqs.weight", (8,), True)
    if per_layer:
        tensor("per_layer_token_embd.weight", (vocab, per_layer * layers))
        tensor("per_layer_model_proj.weight", (per_layer * layers, d))
        tensor("per_layer_proj_norm.weight", (per_layer,), True)
    for layer in range(layers):
        p = f"blk.{layer}."
        swa = layer % 2 == 0
        head, kv = (8, 2) if swa else (16, 1)
        for name in ("attn_norm", "ffn_norm", "post_attention_norm", "post_ffw_norm"):
            tensor(p + name + ".weight", (d,), True)
        tensor(p + "attn_q.weight", (heads * head, d))
        tensor(p + "attn_k.weight", (kv * head, d))
        if swa:
            tensor(p + "attn_v.weight", (kv * head, d))
        for name in ("q", "k"):
            tensor(p + f"attn_{name}_norm.weight", (head,), True)
        tensor(p + "attn_output.weight", (d, heads * head))
        for name, shape in (("gate", (ff, d)), ("up", (ff, d)), ("down", (d, ff))):
            tensor(p + f"ffn_{name}.weight", shape)
        tensor(p + "layer_output_scale.weight", (1,), True)
        if per_layer:
            tensor(p + "inp_gate.weight", (per_layer, d))
            tensor(p + "proj.weight", (d, per_layer))
            tensor(p + "post_norm.weight", (d,), True)
        if opts.get("moe"):
            for name in ("pre_ffw_norm_2", "post_ffw_norm_1", "post_ffw_norm_2"):
                tensor(p + name + ".weight", (d,), True)
            tensor(p + "ffn_gate_inp.weight", (4, d))
            tensor(p + "ffn_gate_inp.scale", (d,), True)
            tensor(p + "ffn_gate_up_exps.weight", (4, 48, d))
            tensor(p + "ffn_down_exps.weight", (4, d, 24))
            tensor(p + "ffn_down_exps.scale", (4,), True)
    finish(w)


def write_model(path, arch, opts):
    arch = opts.get("architecture", arch)
    if arch == "gemma4":
        return write_gemma4_model(path, opts)
    if arch in ("qwen35", "qwen35moe"):
        return write_qwen35_model(path, arch, opts.get("fused_experts", False), opts.get("f16", ()))
    if arch in ("mamba2", "nemotron_h"):
        return write_mamba2_model(path, opts, arch)
    w = gguf.GGUFWriter(path, arch)
    d, heads, head, ff, vocab = opts.get("embedding", 32), 4, 8, 48, 32
    kv = 1 if opts.get("mqa") else heads if opts.get("full_qk") or arch == "deepseek2-ocr" else 2
    layers = opts.get("layers", 2)
    w.add_context_length(128)
    w.add_embedding_length(d)
    w.add_block_count(layers + opts.get("nextn", 0))
    if opts.get("nextn"):
        w.add_nextn_predict_layers(opts["nextn"])
    w.add_feed_forward_length(ff)
    w.add_head_count(heads)
    w.add_head_count_kv(kv)
    w.add_key_length(head)
    w.add_value_length(head)
    w.add_rope_dimension_count(opts.get("rope_dims", head))
    w.add_rope_freq_base(10000.0 if arch == "deepseek2-ocr" else 100.0)
    if opts.get("layer_norm"):
        w.add_layer_norm_eps(1e-5)
    else:
        w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(vocab)
    w.add_tokenizer_model("none")
    if opts.get("embedding_output"):
        w.add_uint32(arch + ".pooling_type", opts.get("pooling", 0))
        w.add_bool(arch + ".attention.causal", opts.get("causal", True))
    if opts.get("linear"):
        w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        w.add_rope_scaling_factor(opts["linear"])
        w.add_rope_freq_base_swa(100.0)
    if opts.get("yarn"):
        w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        w.add_rope_scaling_factor(opts["yarn"])
        w.add_rope_scaling_orig_ctx_len(opts["original_context"])
        w.add_rope_scaling_yarn_beta_fast(opts.get("yarn_beta_fast", 32.0))
        w.add_rope_scaling_yarn_beta_slow(1.0)
        w.add_rope_scaling_yarn_log_mul(opts["yarn_log_mul"])
        w.add_attn_temperature_scale(opts.get("temperature", 0.0))
    if opts.get("swa"):
        w.add_sliding_window(2)
        w.add_sliding_window_pattern(2)
        if arch == "plamo3":
            w.add_rope_freq_base_swa(100.0)
    if opts.get("softcap"):
        w.add_attn_logit_softcapping(2.0)
        w.add_final_logit_softcapping(3.0)
    if arch == "muse-glimmer":
        w.add_logit_scale(1.0)
    if opts.get("moe"):
        w.add_expert_count(4)
        w.add_expert_used_count(2)
        w.add_expert_feed_forward_length(ff)
        w.add_leading_dense_block_count(opts.get("lead", 0))
        if arch != "ernie4_5-moe":
            w.add_expert_shared_count(int(opts.get("shared", False)))
        w.add_expert_shared_feed_forward_length(ff if opts.get("shared") else 0)
        w.add_uint32(arch + ".expert_gating_func", 2 if opts.get("sigmoid") else 1)
        if opts.get("expert_scale"):
            w.add_expert_weights_scale(opts["expert_scale"])
        if arch not in ("hunyuan-moe", "qwen3moe", "ernie4_5-moe", "mellum", "minimax-m2"):
            w.add_expert_weights_norm(arch != "olmoe")
        if arch == "bailingmoe2":
            w.add_expert_group_count(2)
            w.add_expert_group_used_count(1)
        if arch == "ernie4_5-moe":
            w.add_interleave_moe_layer_step(2)
    rng = np.random.default_rng(20260907)
    tensor = tensor_writer(w, rng)

    tensor("token_embd.weight", (vocab, d))
    tensor("output_norm.weight", (d,), True)
    if opts.get("layer_norm"):
        tensor("output_norm.bias", (d,))
    if not opts.get("tied"):
        tensor("output.weight", (vocab, d))
    for layer in range(layers):
        p = f"blk.{layer}."
        if not opts.get("post_only"):
            for name in ("attn_norm", "post_attention_norm" if opts.get("ffn_post_attn") else "ffn_norm"):
                tensor(p + name + ".weight", (d,), True)
                if opts.get("layer_norm"):
                    tensor(p + name + ".bias", (d,))
        if opts.get("post") or opts.get("post_only"):
            for name in ("post_attention_norm", "post_ffw_norm"):
                tensor(p + name + ("" if opts.get("bare_post") else ".weight"), (d,), True)
        if opts.get("fused"):
            tensor(p + "attn_qkv.weight", ((heads + 2 * kv) * head, d))
        else:
            for name, width in (("q", heads * head), ("k", kv * head), ("v", kv * head)):
                tensor(p + f"attn_{name}.weight", (width, d))
                if opts.get("bias"):
                    tensor(p + f"attn_{name}.bias", (width,))
        tensor(p + "attn_output.weight", (d, heads * head))
        if opts.get("layer_norm"):
            tensor(p + "attn_output.bias", (d,))
        if opts.get("qk") or opts.get("full_qk"):
            for name, width in (("q", heads * head), ("k", kv * head)):
                tensor(p + f"attn_{name}_norm.weight", (width if opts.get("full_qk") else head,), True)
        if opts.get("gate"):
            tensor(p + "attn_gate.weight", (heads * head, d))
        if opts.get("moe") and layer >= opts.get("lead", 0):
            tensor(p + "ffn_gate_inp.weight", (4, d))
            for name, shape in (("gate", (4, ff, d)), ("up", (4, ff, d)), ("down", (4, d, ff))):
                tensor(p + f"ffn_{name}_exps.weight", shape)
            if opts.get("selection_bias"):
                tensor(p + "exp_probs_b.bias", (4,))
            if opts.get("shared"):
                for name, shape in (("gate", (ff, d)), ("up", (ff, d)), ("down", (d, ff))):
                    tensor(p + f"ffn_{name}_shexp.weight", shape)
        elif opts.get("relu_squared"):
            for name, shape in (("up", (ff, d)), ("down", (d, ff))):
                tensor(p + f"ffn_{name}.weight", shape)
                tensor(p + f"ffn_{name}.bias", (shape[0],))
        elif opts.get("fused_ffn"):
            tensor(p + "ffn_up.weight", (2 * ff, d))
            tensor(p + "ffn_down.weight", (d, ff))
        else:
            for name, shape in (("gate", (ff, d)), ("up", (ff, d)), ("down", (d, ff))):
                tensor(p + f"ffn_{name}.weight", shape)
    if opts.get("nextn"):
        p = f"blk.{layers}."
        for name, width in (("q", heads * head), ("k", kv * head), ("v", kv * head)):
            tensor(p + f"attn_{name}.weight", (width, d))
        tensor(p + "attn_output.weight", (d, heads * head))
        for name in ("attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"):
            tensor(p + name + ".weight", (head if "_q_" in name or "_k_" in name else d,), True)
        for name, shape in (("gate", (ff, d)), ("up", (ff, d)), ("down", (d, ff))):
            tensor(p + f"ffn_{name}.weight", shape)
        tensor(p + "nextn.eh_proj.weight", (d, 2 * d))
        for name in ("enorm", "hnorm"):
            tensor(p + "nextn." + name + ".weight", (d,), True)
    finish(w)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--embedding-oracle")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "test_data/arch_accuracy")
    parser.add_argument("--architectures", nargs="+", default=list(CASES))
    args = parser.parse_args()
    if not args.embedding_oracle and any(CASES[arch].get("embedding_output") for arch in args.architectures):
        parser.error("--embedding-oracle is required to generate embedding fixtures")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for arch in args.architectures:
            model = Path(tmp) / f"{arch}.gguf"
            reference = model.with_suffix(".bin")
            write_model(model, arch, CASES[arch])
            oracle = args.embedding_oracle if CASES[arch].get("embedding_output") else args.oracle
            result = subprocess.run([oracle, str(model), str(reference)], capture_output=True)
            if result.returncode:
                raise RuntimeError(f"{arch}: oracle failed\n{result.stderr.decode()}")
            if CASES[arch].get("embedding_output"):
                dim, count = np.fromfile(reference, dtype="<i4", count=2)
                values = np.fromfile(reference, dtype="<f4", offset=8 + 4 * count).reshape(count, dim)
                pool = CASES[arch].get("pooling", 0)
                expected = values.mean(axis=0, keepdims=True) if pool == 1 else values[:1] if pool == 2 else values[-1:] if pool == 3 else values
                save_npz(args.out_dir / f"{arch}.npz", {"model": np.fromfile(model, dtype=np.uint8), "embeddings": expected})
                print(f"{arch}: embeddings {expected.shape}", flush=True)
                continue
            vocab = np.fromfile(reference, dtype="<i4", count=1)[0]
            logits = np.fromfile(reference, dtype="<f4", offset=4).reshape(3, vocab)
            assert np.isfinite(logits).all() and np.linalg.norm(logits) > 0
            save_npz(args.out_dir / f"{arch}.npz", {"model": np.fromfile(model, dtype=np.uint8), "logits": logits})
            print(f"{arch}: {model.stat().st_size} bytes, reference logits {logits.shape}", flush=True)


if __name__ == "__main__":
    main()
