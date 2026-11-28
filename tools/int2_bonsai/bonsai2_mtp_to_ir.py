# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build the MTP draft IR (openvino_mtp_model.xml) of a Bonsai 2 model for
OpenVINO GenAI speculative decoding.

prism-ml ships the 4B and 9B as "PQ2_0 + MTP head (Q8_0)" GGUFs: the ternary
model plus one Qwen3.5 multi-token-prediction block (blk.<n_layer>.nextn.*,
a full-attention decoder layer with its own fc/norms, Q8_0). For the 27B the
head comes separately as HF safetensors (mtp.*, BF16; ProCreations/
Ternary-Bonsai-2-27B-MTP), passed with --mtp. GenAI runs the block as the
draft of the main model: it exposes the input of the main lm_head MatMul as
`last_hidden_state`, feeds it to the draft as `hidden_states`, and grafts the
main lm_head weight on the draft's `last_hidden_state`.

The graph comes from the optimum-intel export of a Qwen3.5 model of the same
size (openvino_mtp_model.xml) and every weight is swapped:

  * matrices: int8 with one fp16 scale per 32 inputs, i.e. Q8_0. GGUF blocks
    are copied as they are, BF16 safetensors are quantised that way;
  * norms: the template stores 1 + w, as the GGUF does (safetensors hold w).

Bonsai 2 rotates the lm_head input (bonsai2_gguf_to_ir.py): the main model
exposes H (s * h), with the sign flip s folded into its final norm and H the
blockwise normalised Hadamard transform. The MTP block is in the plain basis
and both ends of the draft are adapted, so that the hidden state it feeds back
for the next draft step stays in the main model's convention:

    hidden_states -> H -> pre_fc_norm_hidden (weight * s)    (H H = I)
    ... -> norm (shared_head_norm * s) -> H -> last_hidden_state

Both H run on the activations. GenAI puts a row Gather in front of the grafted
lm_head, so the GPU plugin cannot fuse the output one into it
(FuseHadamardIntoFC) and it runs as a MatMul on every draft step. For blocks
from 2048 on (9B: 4096) H is built as the Kronecker product of two small
Hadamards instead of a dense block x block MatMul, which drops a 64 MB
constant read per rotation.

    python bonsai2_mtp_to_ir.py \
        --gguf <Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf> \
        --template <qwen35-4b export>/openvino_mtp_model.xml \
        --out-dir <bonsai2-4b genai dir> [--gguf-py <prism llama.cpp>/gguf-py]

    # 27B: the plain GGUF gives the rotation, the head its own file
    python bonsai2_mtp_to_ir.py --gguf <Ternary-Bonsai-2-27B-PQ2_0.gguf> \
        --mtp <Ternary-Bonsai-2-27B-MTP>/model_mtp.safetensors \
        --template <qwen3.5-27b export>/openvino_mtp_model.xml --out-dir <bonsai2-27b genai dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

from bonsai2_gguf_to_ir import find_gguf_py, gguf_f32, hadamard, replace_const

Q8_GROUP = 32
KRON_MIN_BLOCK = 2048
Q8_BLOCK_BYTES = 2 + Q8_GROUP

# template MatMul (friendly name) -> (GGUF stem in the MTP block, safetensors key)
MATMULS = {
    "__module.fc/ov_ext::linear/MatMul": ("nextn.eh_proj", "mtp.fc"),
    "__module.layers.0.self_attn.q_proj/ov_ext::linear/MatMul": ("attn_q", "mtp.layers.0.self_attn.q_proj"),
    "__module.layers.0.self_attn.k_proj/ov_ext::linear/MatMul": ("attn_k", "mtp.layers.0.self_attn.k_proj"),
    "__module.layers.0.self_attn.v_proj/ov_ext::linear/MatMul": ("attn_v", "mtp.layers.0.self_attn.v_proj"),
    "__module.layers.0.self_attn.o_proj/ov_ext::linear/MatMul": ("attn_output", "mtp.layers.0.self_attn.o_proj"),
    "__module.layers.0.mlp.gate_proj/ov_ext::linear/MatMul": ("ffn_gate", "mtp.layers.0.mlp.gate_proj"),
    "__module.layers.0.mlp.up_proj/ov_ext::linear/MatMul": ("ffn_up", "mtp.layers.0.mlp.up_proj"),
    "__module.layers.0.mlp.down_proj/ov_ext::linear/MatMul": ("ffn_down", "mtp.layers.0.mlp.down_proj"),
}
# template norm Multiply (weight on port 1) -> (GGUF stem, safetensors key)
NORMS = {
    "__module.pre_fc_norm_embedding/aten::mul/Multiply_1": ("nextn.enorm", "mtp.pre_fc_norm_embedding"),
    "__module.pre_fc_norm_hidden/aten::mul/Multiply_1": ("nextn.hnorm", "mtp.pre_fc_norm_hidden"),
    "__module.layers.0.input_layernorm/aten::mul/Multiply_1": ("attn_norm", "mtp.layers.0.input_layernorm"),
    "__module.layers.0.post_attention_layernorm/aten::mul/Multiply_1":
        ("post_attention_norm", "mtp.layers.0.post_attention_layernorm"),
    "__module.layers.0.self_attn.q_norm/aten::mul/Multiply_1": ("attn_q_norm", "mtp.layers.0.self_attn.q_norm"),
    "__module.layers.0.self_attn.k_norm/aten::mul/Multiply_1": ("attn_k_norm", "mtp.layers.0.self_attn.k_norm"),
    "__module.norm/aten::type_as/ConvertLike": ("nextn.shared_head_norm", "mtp.norm"),
}
SIGNED_NORMS = ("nextn.hnorm", "nextn.shared_head_norm")


def gguf_q8(t):
    """Q8_0 tensor -> int8 codes [N, K/32, 32] and fp16 scales [N, K/32, 1]."""
    n, k = (int(x) for x in reversed(t.shape))
    raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
    if t.tensor_type.name != "Q8_0" or raw.size != n * (k // Q8_GROUP) * Q8_BLOCK_BYTES:
        sys.exit(f"{t.name}: expected Q8_0 [{n}, {k}], got {t.tensor_type.name}")
    blocks = raw.reshape(n, k // Q8_GROUP, Q8_BLOCK_BYTES)
    scale = np.ascontiguousarray(blocks[:, :, :2]).view("<f2").reshape(n, k // Q8_GROUP, 1)
    return np.ascontiguousarray(blocks[:, :, 2:]).view(np.int8), scale


def quantize_q8(w: np.ndarray):
    """fp32 [N, K] -> Q8_0 as llama.cpp rounds it (scale = absmax / 127 in fp16)."""
    n, k = w.shape
    g = w.reshape(n, k // Q8_GROUP, Q8_GROUP)
    d = np.abs(g).max(axis=2, keepdims=True) / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    return np.clip(np.rint(g * inv), -127, 127).astype(np.int8), d.astype(np.float16)


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """All tensors of a safetensors file as fp32 (numpy has no bfloat16)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        data = np.frombuffer(f.read(), np.uint8)
    out = {}
    for name, h in header.items():
        if name == "__metadata__":
            continue
        raw = data[h["data_offsets"][0]:h["data_offsets"][1]]
        if h["dtype"] == "BF16":
            x = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif h["dtype"] in ("F16", "F32"):
            x = raw.view(np.float16 if h["dtype"] == "F16" else np.float32).astype(np.float32)
        else:
            sys.exit(f"{path}: {name} has unsupported dtype {h['dtype']}")
        out[name] = x.reshape(h["shape"])
    return out


def q8_weight(q: np.ndarray, scale: np.ndarray, et: ov.Type) -> ov.Output:
    """int8 [N, K/32, 32] + fp16 scales -> [N, K] weight subgraph."""
    n, k = q.shape[0], q.shape[1] * q.shape[2]
    x = ops.multiply(ops.convert(ops.constant(q), ov.Type.f16), ops.constant(scale))
    x = ops.reshape(x, ops.constant(np.array([n, k], np.int64)), False)
    return ops.convert(x, et).output(0)


def norm_const(node):
    src = node.input_value(1).get_node()
    while src.get_type_name() == "Convert":
        src = src.input_value(0).get_node()
    return src if src.get_type_name() == "Constant" else None


def rotation(block: int):
    """x -> x H_block on each block of the last dim. From 2048 on, H_block is applied
    as H_a (x) H_b, a * b = block (Y = H_a X H_b on the [a, b] view of a block), so
    the graph holds two small constants instead of a dense block x block one."""
    if block < KRON_MIN_BLOCK:
        H = ops.constant(hadamard(block).astype(np.float32))
        H.set_friendly_name(f"hadamard.H{block}")
    else:
        a = 1 << (block.bit_length() - 1) // 2
        b = block // a
        Ha = ops.constant(hadamard(a).astype(np.float32))
        Hb = ops.constant(hadamard(b).astype(np.float32))
        Ha.set_friendly_name(f"hadamard.H{a}")
        Hb.set_friendly_name(f"hadamard.H{b}")

    def rotate(x: ov.Output) -> ov.Output:
        k = x.get_partial_shape()[-1].get_length()
        if block < KRON_MIN_BLOCK:
            r = ops.reshape(x, ops.constant(np.array([0, 0, k // block, block], np.int64)), True)
            h = ops.matmul(r, H, False, False)
        else:
            r = ops.reshape(x, ops.constant(np.array([0, 0, k // block, a, b], np.int64)), True)
            h = ops.matmul(Ha, ops.matmul(r, Hb, False, False), False, False)
        return ops.reshape(h, ops.constant(np.array([0, 0, k], np.int64)), True).output(0)
    return rotate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True,
                    help="Ternary-Bonsai-2-<size>-PQ2_0-MTP-Q8_0.gguf, or the plain PQ2_0 GGUF with --mtp")
    ap.add_argument("--mtp", help="MTP head as HF safetensors (mtp.* keys), instead of the GGUF block")
    ap.add_argument("--template", required=True, help="optimum-intel openvino_mtp_model.xml of Qwen3.5-<size>")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gguf-py")
    a = ap.parse_args()

    sys.path.insert(0, find_gguf_py(a.gguf_py))
    from gguf import GGUFReader  # noqa: E402

    t0 = time.perf_counter()
    r = GGUFReader(a.gguf)
    fields = {k: v.contents() for k, v in r.fields.items()}
    if fields.get("general.architecture") != "qwen35":
        sys.exit(f"unsupported architecture {fields.get('general.architecture')!r}")
    nextn = int(fields.get("qwen35.nextn_predict_layers", 0))
    il = int(fields["qwen35.block_count"]) - nextn
    hidden = int(fields["qwen35.embedding_length"])
    tensors = {t.name: t for t in r.tensors}

    if a.mtp:
        st = read_safetensors(a.mtp)

        def matrix(stem, key):
            return quantize_q8(st[key + ".weight"])

        def norm(stem, key):
            return 1.0 + st[key + ".weight"]
        source = f"{a.mtp}: {len(st)} tensors"
    else:
        if nextn != 1:
            sys.exit(f"expected one MTP block, GGUF has nextn_predict_layers={nextn} (or pass --mtp)")

        def tensor(stem):
            t = tensors.get(f"blk.{il}.{stem}.weight")
            if t is None:
                sys.exit(f"blk.{il}.{stem}.weight missing from GGUF")
            return t

        def matrix(stem, key):
            return gguf_q8(tensor(stem))

        def norm(stem, key):
            return gguf_f32(tensor(stem))
        source = f"MTP block blk.{il}"

    # The draft must match the main model's lm_head input convention.
    if int(fields.get("prism.hadamard.version", 0)) not in (1, 2):
        sys.exit("GGUF has no prism.hadamard contract")
    head = "output.weight" if "output.weight" in tensors else "token_embd.weight"
    tied = bool(fields.get("prism.hadamard.tied_output", False))
    if head not in fields["prism.hadamard.weight_names"] and not tied:
        sys.exit("the lm_head is not in the rotated basis")
    if any(n.startswith(f"blk.{il}.") for n in fields["prism.hadamard.weight_names"]):
        sys.exit(f"MTP block blk.{il} unexpectedly rotated")
    block = int(fields["prism.hadamard.block_size"])
    vals = np.asarray(fields["prism.hadamard.sign_values"], dtype=np.float32)
    s, off = None, 0
    for w in fields["prism.hadamard.sign_widths"]:
        if int(w) == hidden:
            s = vals[off:off + hidden]
        off += int(w)
    if s is None or not np.isin(s, [-1, 1]).all() or hidden % block:
        sys.exit(f"no valid {hidden}-wide sign vector / H{block}")
    print(f"[mtp] {a.gguf}: hidden {hidden}, H{block}; head from {source}")

    model = ov.Core().read_model(a.template)
    nodes = {n.get_friendly_name(): n for n in model.get_ordered_ops()}

    for name, (stem, key) in MATMULS.items():
        mm = nodes.get(name)
        if mm is None:
            sys.exit(f"{name} not in template")
        w = q8_weight(*matrix(stem, key), mm.get_input_element_type(1))
        want = mm.get_input_partial_shape(1)
        if list(w.get_shape()) != [d.get_length() for d in want]:
            sys.exit(f"{name}: MTP head {list(w.get_shape())} vs template {want}")
        mm.input(1).replace_source_output(w)
    for name, (stem, key) in NORMS.items():
        node = nodes.get(name)
        c = norm_const(node) if node is not None else None
        if c is None:
            sys.exit(f"{name}: norm weight not found in template")
        x = norm(stem, key).reshape(-1)
        if stem in SIGNED_NORMS:
            x = x * s
        replace_const(c, x.reshape(c.get_output_shape(0)), ov.Type.f32)

    rotate = rotation(block)
    hs = next((p for p in model.get_parameters() if "hidden_states" in p.output(0).get_names()), None)
    if hs is None:
        sys.exit("template has no hidden_states input")
    consumers = list(hs.output(0).get_target_inputs())
    rot_in = rotate(hs.output(0))
    for c in consumers:
        c.replace_source_output(rot_in)
    res = model.get_results()
    if len(res) != 1 or "last_hidden_state" not in model.outputs[0].get_names():
        sys.exit("template must have a single last_hidden_state output")
    src = res[0].input_value(0)
    rot_out = rotate(src)
    rot_out.set_names(src.get_names())
    src.set_names(set())
    res[0].input(0).replace_source_output(rot_out)
    model.validate_nodes_and_infer_types()

    os.makedirs(a.out_dir, exist_ok=True)
    ov.save_model(model, os.path.join(a.out_dir, "openvino_mtp_model.xml"), compress_to_fp16=False)
    print(f"[mtp] wrote {a.out_dir}/openvino_mtp_model.xml ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
