# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Generate small GGUF models and logits from real llama.cpp CPU, never from OpenVINO.

Usage: PYTHONPATH=<llama.cpp>/gguf-py python gen_arch_accuracy.py --oracle <architecture_oracle>
Build architecture_oracle.cpp against a CPU-only llama.cpp (include/, ggml/include/,
link libllama, libggml, libggml-base). The generated NPZs are consumed without llama.cpp.
"""
import argparse
import io
import zipfile
from pathlib import Path
import subprocess
import tempfile

import gguf
import numpy as np

CASES = {
    "llama": {}, "qwen2": {"bias": True}, "qwen3": {"qk": True},
    "phi3": {"fused": True, "fused_ffn": True}, "minicpm": {},
    "olmoe": {"moe": True, "full_qk": True},
    "hunyuan-dense": {"qk": True}, "hunyuan-moe": {"qk": True, "moe": True, "shared": True},
    "qwen3moe": {"qk": True, "moe": True},
    "gemma": {"mqa": True, "tied": True},
    "gemma2": {"post": True, "swa": True, "tied": True, "softcap": True},
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


def write_model(path, arch, opts):
    arch = opts.get("architecture", arch)
    w = gguf.GGUFWriter(path, arch)
    d, heads, head, ff, vocab = opts.get("embedding", 32), 4, 8, 48, 32
    kv = 1 if opts.get("mqa") else heads if opts.get("full_qk") or arch == "deepseek2-ocr" else 2
    layers = opts.get("layers", 2)
    w.add_context_length(128)
    w.add_embedding_length(d)
    w.add_block_count(layers)
    w.add_feed_forward_length(ff)
    w.add_head_count(heads)
    w.add_head_count_kv(kv)
    w.add_key_length(head)
    w.add_value_length(head)
    w.add_rope_dimension_count(head)
    w.add_rope_freq_base(10000.0 if arch == "deepseek2-ocr" else 100.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(vocab)
    w.add_tokenizer_model("none")
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
        if arch not in ("qwen3moe", "ernie4_5-moe", "mellum"):
            w.add_expert_weights_norm(arch != "olmoe")
        if arch == "bailingmoe2":
            w.add_expert_group_count(2)
            w.add_expert_group_used_count(1)
        if arch == "ernie4_5-moe":
            w.add_interleave_moe_layer_step(2)
    rng = np.random.default_rng(20260907)

    def tensor(name, shape, norm=False):
        values = rng.uniform(-1, 1, shape).astype(np.float32)
        values = (1 + values * .3) if norm else values * .2
        w.add_tensor(name, values)

    tensor("token_embd.weight", (vocab, d))
    tensor("output_norm.weight", (d,), True)
    if not opts.get("tied"):
        tensor("output.weight", (vocab, d))
    for layer in range(layers):
        p = f"blk.{layer}."
        if not opts.get("post_only"):
            for name in ("attn_norm", "ffn_norm"):
                tensor(p + name + ".weight", (d,), True)
        if opts.get("post") or opts.get("post_only"):
            for name in ("post_attention_norm", "post_ffw_norm"):
                tensor(p + name + ".weight", (d,), True)
        if opts.get("fused"):
            tensor(p + "attn_qkv.weight", ((heads + 2 * kv) * head, d))
        else:
            for name, width in (("q", heads * head), ("k", kv * head), ("v", kv * head)):
                tensor(p + f"attn_{name}.weight", (width, d))
                if opts.get("bias"):
                    tensor(p + f"attn_{name}.bias", (width,))
        tensor(p + "attn_output.weight", (d, heads * head))
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
        elif opts.get("fused_ffn"):
            tensor(p + "ffn_up.weight", (2 * ff, d))
            tensor(p + "ffn_down.weight", (d, ff))
        else:
            for name, shape in (("gate", (ff, d)), ("up", (ff, d)), ("down", (d, ff))):
                tensor(p + f"ffn_{name}.weight", shape)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "test_data/arch_accuracy")
    parser.add_argument("--architectures", nargs="+", default=list(CASES))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for arch in args.architectures:
            model = Path(tmp) / f"{arch}.gguf"
            reference = model.with_suffix(".bin")
            write_model(model, arch, CASES[arch])
            result = subprocess.run([args.oracle, str(model), str(reference)], capture_output=True)
            if result.returncode:
                raise RuntimeError(f"{arch}: oracle failed\n{result.stderr.decode()}")
            vocab = np.fromfile(reference, dtype="<i4", count=1)[0]
            logits = np.fromfile(reference, dtype="<f4", offset=4).reshape(3, vocab)
            assert np.isfinite(logits).all() and np.linalg.norm(logits) > 0
            # cnpy reads standard ZIP headers; NumPy's savez forces ZIP64 even for tiny arrays.
            with zipfile.ZipFile(args.out_dir / f"{arch}.npz", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for key, array in {"model": np.fromfile(model, dtype=np.uint8), "logits": logits}.items():
                    payload = io.BytesIO()
                    np.save(payload, array)
                    info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, payload.getvalue())
            print(f"{arch}: {model.stat().st_size} bytes, reference logits {logits.shape}", flush=True)


if __name__ == "__main__":
    main()
