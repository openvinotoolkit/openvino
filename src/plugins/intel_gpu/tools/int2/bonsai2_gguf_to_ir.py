#!/usr/bin/env python3
"""Build the Bonsai 2 27B u2 IR from its PQ2_0 GGUF, using the Bonsai 1 27B IR
as a template.

Bonsai 2 27B (prism-ml/Ternary-Bonsai-2-27B-gguf) has the Qwen3.5-27B
architecture of Bonsai 27B, so the exported+quantized Bonsai 1 IR already holds
the right graph. This tool keeps that graph and swaps every weight:

  * ternary MatMuls: the PQ2_0 blocks *are* the u2 layout the GPU plugin's
    weight-compression pattern expects (codes {0,1,2}, zero point 1, one fp16
    scale per 128), so the code bytes are copied and only the scales are split
    off;
  * dense tensors (norms, GDN A_log/dt_bias/conv1d/in_proj_a/in_proj_b): read
    from the GGUF, with llama.cpp's V-head reorder undone the way the HF model
    expects;
  * the token embedding goes into the separate text-embeddings IR, still ternary
    (u8-packed codes + fp16 scales, 0.34 GB): the device gathers the rows and
    dequantises them.

Bonsai 2 additionally stores its folded weights in a rotated basis: the input
of each folded projection is sign-flipped and put through a blockwise (1024)
normalised Walsh-Hadamard transform, and the embedding table is stored rotated.
The transform is inserted into the graph (Reshape -> MatMul with one shared
H_1024 constant -> Reshape); the sign flips are folded into the preceding
RMSNorm weight where every consumer is folded (compensating the dense GDN gate
projections), into the up_proj rows for down_proj, and kept as an explicit
Multiply only for the two attention output projections whose producer is not a
per-channel weight. The embedding rows get the inverse transform after the
gather, on the device.

    python bonsai2_gguf_to_ir.py \
        --gguf <Ternary-Bonsai-2-27B-PQ2_0.gguf> \
        --template-dir <bonsai27b-u2 dir with openvino_model.xml> \
        --out-dir <bonsai2-27b-u2> [--gguf-py <prism llama.cpp>/gguf-py]

    # self-check: run it on the Bonsai 1 GGUF and compare against its own IR
    python bonsai2_gguf_to_ir.py --gguf <Ternary-Bonsai-27B-PQ2_0.gguf> \
        --template-dir ... --template-embed ... --verify

Reading PQ2_0 needs the PrismML llama.cpp fork's gguf-py (type id 142).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

GROUP = 128
BLOCK_BYTES = 2 + GROUP // 4
PREFIX = "__module.model."          # optimum's module prefix in node names
LM = "model.language_model."

# HF module leaf -> GGUF tensor stem (per block)
QUANT_MAP = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
    "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.out_proj": "ssm_out",
    "linear_attn.in_proj_a": "ssm_alpha", "linear_attn.in_proj_b": "ssm_beta",
}
# node-name suffix (relative to the layer) -> (GGUF stem, kind)
DENSE_MAP = {
    "input_layernorm/aten::mul/Multiply_1": ("attn_norm.weight", "norm"),
    "post_attention_layernorm/aten::mul/Multiply_1": ("post_attention_norm.weight", "norm"),
    "self_attn.q_norm/aten::mul/Multiply_1": ("attn_q_norm.weight", "norm"),
    "self_attn.k_norm/aten::mul/Multiply_1": ("attn_k_norm.weight", "norm"),
    "linear_attn.norm/aten::mul/Multiply_1": ("ssm_norm.weight", "norm"),
    "linear_attn/aten::exp/Exp": ("ssm_a", "a_log"),
    "linear_attn/aten::add/Add_2": ("ssm_dt.bias", "dt_bias"),
    "linear_attn/aten::_convolution/GroupConvolution": ("ssm_conv1d.weight", "conv1d"),
}


def find_gguf_py(explicit: str | None) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [explicit or "", os.environ.get("PRISM_GGUF_PY", ""),
             os.path.join(here, "..", "..", "..", "..", "..", "..", "third_party",
                          "llama.cpp-prism", "gguf-py")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "gguf", "constants.py")):
            with open(os.path.join(c, "gguf", "constants.py")) as fh:
                if "PQ2_0" in fh.read():
                    return os.path.abspath(c)
    sys.exit("need the PrismML llama.cpp fork's gguf-py (knows PQ2_0): --gguf-py or "
             "$PRISM_GGUF_PY; git clone --depth 1 -b prism "
             "https://github.com/PrismML-Eng/llama.cpp")


def gguf_f32(t) -> np.ndarray:
    shape = tuple(int(x) for x in reversed(t.shape))
    arr = np.asarray(t.data)
    if t.tensor_type.name == "BF16":
        return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if t.tensor_type.name in ("F32", "F16"):
        return arr.astype(np.float32).reshape(shape)
    sys.exit(f"{t.name}: unsupported dense type {t.tensor_type.name}")


def pq2_blocks(t) -> np.ndarray:
    """PQ2_0 tensor -> uint8 [N, K/128, 34] (fp16 scale, then 32 code bytes)."""
    n, k = (int(x) for x in reversed(t.shape))
    raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
    if raw.size != n * (k // GROUP) * BLOCK_BYTES:
        sys.exit(f"{t.name}: PQ2_0 byte length mismatch")
    return raw.reshape(n, k // GROUP, BLOCK_BYTES)


def blocks_dequant(blocks: np.ndarray) -> np.ndarray:
    """[N, G, 34] -> fp32 [N, G*128]."""
    n, g, _ = blocks.shape
    scale = blocks[:, :, :2].copy().view("<f2").astype(np.float32).reshape(n, g, 1)
    q = blocks[:, :, 2:]
    codes = np.stack([(q >> (2 * j)) & 3 for j in range(4)], axis=-1)  # [N,G,32,4]
    codes = codes.reshape(n, g, GROUP).astype(np.float32) - 1.0
    return (codes * scale).reshape(n, g * GROUP)


def flip_code_bytes(q: np.ndarray) -> np.ndarray:
    """Negate ternary codes c -> 2 - c in every 2-bit lane (0<->2, 1 stays)."""
    q = q.astype(np.uint8)
    hi = (~(q | (q << 1))) & 0xAA
    return (hi | (q & 0x55)).astype(np.uint8)


def hadamard(n: int) -> np.ndarray:
    i = np.arange(n)
    pop = np.array([bin(x).count("1") for x in range(n)])
    return np.where(pop[i[:, None] & i[None, :]] % 2, -1.0, 1.0) / np.sqrt(n)


def ternary_embedding(blocks: np.ndarray, block: int, signs_k: np.ndarray | None) -> ov.Model:
    """Embedding model that gathers the ternary rows and dequantises them on the device.

    Byte j of a u8 table row holds the codes of columns j, j+K/4, j+K/2 and j+3K/4 in its
    four 2-bit lanes, so unpacking is four floor/mod planes concatenated in column order.
    With a Hadamard contract the gathered rows get the inverse rotation (x H_block * signs)."""
    v, g, _ = blocks.shape
    k = g * GROUP
    q = blocks[:, :, 2:]
    codes = np.stack([(q >> (2 * j)) & 3 for j in range(4)], axis=-1).reshape(v, k)
    quarter = k // 4
    table = np.zeros((v, quarter), np.uint8)
    for p in range(4):
        table |= codes[:, p * quarter:(p + 1) * quarter] << (2 * p)
    del codes
    scales = np.ascontiguousarray(blocks[:, :, :2]).view("<f2").reshape(v, g)

    ids = ops.parameter([-1, -1], ov.Type.i64, name="input")
    idx = ops.convert(ids, ov.Type.i32)
    axis = ops.constant(np.int32(0))
    qf = ops.convert(ops.gather(ops.constant(table), idx, axis), ov.Type.f32)
    planes = []
    for p in range(4):
        x = ops.floor(ops.divide(qf, ops.constant(np.float32(4 ** p)))) if p else qf
        planes.append(ops.floor_mod(x, ops.constant(np.float32(4))) if p < 3 else x)
    w = ops.subtract(ops.concat(planes, -1), ops.constant(np.float32(1)))    # codes {0,1,2} - zp
    s = ops.convert(ops.gather(ops.constant(scales), idx, axis), ov.Type.f32)
    w = ops.multiply(ops.reshape(w, ops.constant(np.array([0, 0, g, GROUP], np.int64)), True),
                     ops.unsqueeze(s, ops.constant(np.int64(-1))))
    w = ops.reshape(w, ops.constant(np.array([0, 0, k], np.int64)), True)
    if signs_k is not None:
        w = ops.reshape(w, ops.constant(np.array([0, 0, k // block, block], np.int64)), True)
        w = ops.matmul(w, ops.constant(hadamard(block).astype(np.float32)), False, False)
        w = ops.multiply(ops.reshape(w, ops.constant(np.array([0, 0, k], np.int64)), True),
                         ops.constant(signs_k.reshape(1, 1, k).astype(np.float32)))
    out = ops.result(w)
    out.output(0).get_tensor().set_names({"inputs_embeds"})
    return ov.Model([out], [ids], "text_embeddings")


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16)


def vperm(nv: int, nk: int, unit: int) -> np.ndarray:
    """llama.cpp tiled V order -> HF grouped order (index vector)."""
    return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)


def const_of(node, port: int):
    """Constant behind input `port` of node, looking through Convert/Reshape."""
    src = node.input_value(port).get_node()
    while src.get_type_name() in ("Convert", "Reshape"):
        src = src.input_value(0).get_node()
    return src if src.get_type_name() == "Constant" else None


def u2_chain(mm):
    """(u2 Constant, scale Constant) of the weight-decompression subgraph on
    MatMul input 1: Convert <- Reshape <- Multiply(Subtract(Convert(u2), zp), scale)."""
    src = mm.input_value(1).get_node()
    mul = None
    while src.get_type_name() in ("Convert", "Reshape", "Multiply", "Subtract"):
        if src.get_type_name() == "Multiply":
            mul = src
        src = src.input_value(0).get_node()
    if src.get_type_name() != "Constant" or src.get_output_element_type(0) != ov.Type.u2 or mul is None:
        return None, None
    return src, const_of(mul, 1)


def replace_const(old, data: np.ndarray, et: ov.Type | None = None):
    if et is None:
        et = old.get_output_element_type(0)
    if et == ov.Type.bf16:
        t = ov.Tensor(et, list(data.shape))
        np.frombuffer(t.data, dtype=np.uint16)[:] = f32_to_bf16_bits(data).reshape(-1)
        new = ops.constant(t)
    else:
        new = ops.constant(data.astype(et.to_dtype()))
    new.set_friendly_name(old.get_friendly_name())
    for tgt in list(old.output(0).get_target_inputs()):
        tgt.replace_source_output(new.output(0))
    return new


def const_f32(c) -> np.ndarray:
    d = c.get_data()
    if c.get_output_element_type(0) == ov.Type.bf16:
        return (d.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return d.astype(np.float32)


class Report:
    def __init__(self):
        self.worst = {}

    def add(self, kind: str, name: str, got: np.ndarray, want: np.ndarray):
        if got.shape != want.shape:
            sys.exit(f"{name}: shape {got.shape} vs template {want.shape}")
        err = float(np.abs(got.astype(np.float32) - want.astype(np.float32)).max())
        rel = err / (float(np.abs(want).max()) + 1e-12)
        if kind not in self.worst or rel > self.worst[kind][0]:
            self.worst[kind] = (rel, err, name)

    def dump(self):
        for k, (rel, err, name) in sorted(self.worst.items()):
            print(f"  {k:10s} worst rel {rel:.3e} (abs {err:.3e}) at {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--template-dir", required=True,
                    help="dir with the Bonsai 1 27B u2 openvino_model.xml")
    ap.add_argument("--template-embed",
                    help="Bonsai 1 27B openvino_text_embeddings_model.xml (--verify only)")
    ap.add_argument("--out-dir")
    ap.add_argument("--gguf-py")
    ap.add_argument("--verify", action="store_true",
                    help="compare GGUF-derived tensors with the template instead of writing")
    ap.add_argument("--no-hadamard", action="store_true",
                    help="ignore the prism.hadamard contract (debug)")
    ap.add_argument("--explicit-signs", action="store_true",
                    help="keep every sign flip as a Multiply instead of folding into weights (debug)")
    ap.add_argument("--no-fold-norm", action="store_true", help="debug: no RMSNorm sign folds")
    ap.add_argument("--no-fold-up", action="store_true", help="debug: no up_proj row folds")
    a = ap.parse_args()
    fold_norm = not (a.explicit_signs or a.no_fold_norm)
    fold_up = not (a.explicit_signs or a.no_fold_up)
    if not a.verify and not a.out_dir:
        ap.error("--out-dir is required unless --verify")
    if a.verify and not a.template_embed:
        ap.error("--verify needs --template-embed")

    sys.path.insert(0, find_gguf_py(a.gguf_py))
    from gguf import GGUFReader  # noqa: E402

    t0 = time.perf_counter()
    r = GGUFReader(a.gguf)
    fields = {k: v.contents() for k, v in r.fields.items()}
    if fields.get("general.architecture") != "qwen35":
        sys.exit(f"unsupported architecture {fields.get('general.architecture')!r}")
    g = lambda key: fields["qwen35." + key]  # noqa: E731
    nv, nk = int(g("ssm.time_step_rank")), int(g("ssm.group_count"))
    hd = int(g("ssm.inner_size")) // nv
    qk_rows = 2 * nk * int(g("ssm.state_size"))
    perm_hd, perm_1 = vperm(nv, nk, hd), vperm(nv, nk, 1)
    # llama.cpp also permutes ssm_out's input columns to its tiled V order unless
    # the exporter kept them grouped (Bonsai 2 must: a folded weight cannot be
    # column-permuted).
    v_grouped = bool(fields.get("prism.hadamard.gdn_v_grouped", False))
    tensors = {t.name: t for t in r.tensors}
    print(f"[ir] {a.gguf}: {len(tensors)} tensors, GDN nv={nv} nk={nk} hd={hd}")

    # ---- Hadamard contract -------------------------------------------------
    folded: set[str] = set()
    signs: dict[int, np.ndarray] = {}
    block = 0
    if int(fields.get("prism.hadamard.version", 0)) == 1 and not a.no_hadamard:
        if fields["prism.hadamard.transform"] != "normalized-sylvester-walsh-hadamard" \
                or fields["prism.hadamard.axis"] != "input-last-dimension" \
                or fields["prism.hadamard.sign_mode"] != "explicit" \
                or not fields.get("prism.hadamard.gdn_v_grouped", False):
            sys.exit("unexpected prism.hadamard contract")
        block = int(fields["prism.hadamard.block_size"])
        folded = set(fields["prism.hadamard.weight_names"])
        inverse = set(fields.get("prism.hadamard.inverse_weight_names", []))
        if inverse != {"token_embd.weight"}:
            sys.exit(f"unexpected inverse set {inverse}")
        vals = np.asarray(fields["prism.hadamard.sign_values"], dtype=np.float32)
        off = 0
        for w in fields["prism.hadamard.sign_widths"]:
            w = int(w)
            signs[w] = vals[off:off + w]
            off += w
        if off != len(vals) or any(not np.isin(s, [-1, 1]).all() for s in signs.values()):
            sys.exit("bad sign vectors")
        print(f"[ir] hadamard H{block}: {len(folded)} folded weights, sign widths {sorted(signs)}")
    else:
        print("[ir] no hadamard contract: plain ternary model")

    core = ov.Core()
    model = core.read_model(os.path.join(a.template_dir, "openvino_model.xml"))
    rep = Report()

    # ---- ternary + dense MatMul weights -------------------------------------
    mm_re = re.compile(re.escape(PREFIX) + r"(.+)/ov_ext::linear/MatMul$")
    matmuls = {}
    for node in model.get_ordered_ops():
        if node.get_type_name() != "MatMul":
            continue
        m = mm_re.match(node.get_friendly_name())
        if m:
            matmuls[m.group(1)] = node
    print(f"[ir] {len(matmuls)} linear MatMuls in template")

    layer_re = re.compile(re.escape(LM) + r"layers\.(\d+)\.(.+)$")
    n_u2 = n_dense = n_row_fold = 0
    k_of = {}               # hf module -> K
    for hf, mm in sorted(matmuls.items()):
        if hf == "lm_head":
            gname, leaf, layer = "output.weight", "lm_head", -1
        else:
            lm = layer_re.match(hf)
            if not lm:
                sys.exit(f"unexpected MatMul {hf}")
            layer, leaf = int(lm.group(1)), lm.group(2)
            if leaf not in QUANT_MAP:
                sys.exit(f"unmapped MatMul {hf}")
            gname = f"blk.{layer}.{QUANT_MAP[leaf]}.weight"
        t = tensors.get(gname)
        if t is None:
            sys.exit(f"{hf}: {gname} missing from GGUF")
        k_of[hf] = int(t.shape[0])
        # row permutation undoing llama.cpp's tiled V heads
        stem = QUANT_MAP.get(leaf)
        rows = cols = None
        if stem == "attn_qkv":
            rows = np.concatenate([np.arange(qk_rows), qk_rows + perm_hd])
        elif stem == "attn_gate":
            rows = perm_hd
        elif stem in ("ssm_alpha", "ssm_beta"):
            rows = perm_1
        elif stem == "ssm_out" and not v_grouped:
            if hd != GROUP:
                sys.exit("ssm_out column permutation needs head_v_dim == 128")
            cols = perm_1                      # whole 128-groups move

        u2c, sc_c = u2_chain(mm)
        if u2c is not None and t.tensor_type.name != "PQ2_0":
            # ternary in the template, dense in this GGUF (Bonsai 2 keeps the
            # GDN gate projections in bf16): drop the decompression subgraph
            if a.verify:
                sys.exit(f"{hf}: template is u2 but GGUF holds {t.tensor_type.name}")
            w = gguf_f32(t)
            if rows is not None:
                w = w[rows]
            if w.shape[1] in signs and fold_norm \
                    and _norm_folded_input(mm, folded, tensors, layer):
                w = w * signs[w.shape[1]][None, :]
            wc = ops.constant(w.astype(np.float16))
            wc.set_friendly_name(u2c.get_friendly_name().replace("_compressed", "") + "_dense")
            mm.input(1).replace_source_output(
                ops.convert(wc, mm.input_value(1).get_element_type()).output(0))
            n_dense += 1
        elif u2c is not None:
            if t.tensor_type.name != "PQ2_0":
                sys.exit(f"{hf}: template is u2 but GGUF holds {t.tensor_type.name}")
            blocks = pq2_blocks(t)
            if rows is not None:
                blocks = blocks[rows]
            if cols is not None:
                blocks = blocks[:, cols]
            n, gcount = blocks.shape[:2]
            if list(u2c.get_output_shape(0)) != [n, gcount, GROUP]:
                sys.exit(f"{hf}: u2 shape {list(u2c.get_output_shape(0))} vs GGUF {[n, gcount, GROUP]}")
            codes = np.ascontiguousarray(blocks[:, :, 2:])
            scale = blocks[:, :, :2].copy().view("<f2").reshape(n, gcount, 1)
            if a.verify:
                same = np.array_equal(u2c.get_data().view(np.uint8), codes.reshape(-1))
                tmpl_scale = sc_c.get_data().astype(np.float32).reshape(n, gcount)
                # groups that are all-zero get scale 0 from the quantizer
                allzero = (codes == 0x55).all(axis=2)
                sd = np.abs(tmpl_scale - scale.reshape(n, gcount).astype(np.float32))
                sd[allzero] = 0
                if not same or sd.max() != 0:
                    sys.exit(f"{hf}: verify failed codes_equal={same} scale_maxdiff={sd.max()}")
            else:
                if leaf == "mlp.up_proj" and f"blk.{layer}.ffn_down.weight" in folded \
                        and fold_up:
                    # down_proj's input sign flip folds into up_proj's output rows
                    neg = signs[n] < 0
                    codes = codes.reshape(n, -1).copy()
                    codes[neg] = flip_code_bytes(codes[neg])
                    n_row_fold += 1
                tq = ov.Tensor(ov.Type.u2, [n, gcount, GROUP])
                np.frombuffer(tq.data, dtype=np.uint8)[:] = codes.reshape(-1)
                new_q = ops.constant(tq)
                new_q.set_friendly_name(u2c.get_friendly_name())
                for tgt in list(u2c.output(0).get_target_inputs()):
                    tgt.replace_source_output(new_q.output(0))
                replace_const(sc_c, scale.astype(np.float16))
            n_u2 += 1
        else:
            # dense fp16/bf16 weight behind a Convert
            wc = const_of(mm, 1)
            if wc is None:
                sys.exit(f"{hf}: no constant weight found")
            w = gguf_f32(t)
            if rows is not None:
                w = w[rows]
            if a.verify:
                rep.add("dense_w", hf, w, const_f32(wc))
            else:
                if gname in folded:
                    sys.exit(f"{hf}: dense weight marked folded")
                # compensate the sign flip folded into its RMSNorm input
                k = w.shape[1]
                if k in signs and fold_norm \
                        and _norm_folded_input(mm, folded, tensors, layer):
                    w = w * signs[k][None, :]
                replace_const(wc, w)
            n_dense += 1
    print(f"[ir] weights: {n_u2} ternary, {n_dense} dense, {n_row_fold} up_proj row folds "
          f"({time.perf_counter() - t0:.0f}s)")

    # ---- dense per-layer tensors --------------------------------------------
    n_layers = int(g("block_count"))
    norm_nodes = {}
    for node in model.get_ordered_ops():
        name = node.get_friendly_name()
        if not name.startswith(PREFIX + LM):
            continue
        rest = name[len(PREFIX + LM):]
        if rest == "norm/aten::mul/Multiply_1":
            layer, suffix = -1, "norm"
            gname, kind = "output_norm.weight", "norm"
        else:
            lm = re.match(r"layers\.(\d+)\.(.+)$", rest)
            if not lm or lm.group(2) not in DENSE_MAP:
                continue
            layer, suffix = int(lm.group(1)), lm.group(2)
            stem, kind = DENSE_MAP[suffix]
            gname = f"blk.{layer}.{stem}"
        t = tensors.get(gname)
        if t is None:
            sys.exit(f"{name}: {gname} missing from GGUF")
        x = gguf_f32(t)
        port = 0 if kind == "a_log" else 1
        if kind == "norm" and suffix == "linear_attn.norm/aten::mul/Multiply_1":
            port = 0
        c = const_of(node, port)
        if c is None:
            sys.exit(f"{name}: constant not found on port {port}")
        shape = list(c.get_output_shape(0))
        if kind == "a_log":
            if not (x < 0).all():
                sys.exit(f"{gname}: expected A = -exp(A_log) < 0")
            x = (-x)[perm_1]                    # template holds exp(A_log)
        elif kind == "dt_bias":
            x = x[perm_1]
        elif kind == "conv1d":
            idx = np.concatenate([np.arange(qk_rows), qk_rows + perm_hd])
            x = x[idx]                          # [C, k]
        x = x.reshape(shape)
        if a.verify:
            rep.add(kind if kind != "norm" else f"norm:{suffix.split('/')[0].split('.')[-1]}",
                    name, x, const_f32(c))
        else:
            c = replace_const(c, x)          # the fold below must see the new data
        if kind == "norm":
            norm_nodes[(layer, suffix.split("/")[0])] = (node, c)
    print(f"[ir] dense per-layer tensors done ({time.perf_counter() - t0:.0f}s)")

    if a.verify:
        rep.dump()

    # ---- Hadamard rotation ---------------------------------------------------
    if folded and not a.verify:
        H = ops.constant(hadamard(block).astype(np.float32))
        H.set_friendly_name("hadamard.H1024")
        n_rot = n_sign_mul = n_norm_fold = 0
        rotated_outputs = {}    # producer output -> rotated output (shared)

        def rotate(src_out, k: int, explicit_sign: bool):
            nonlocal n_rot, n_sign_mul
            key = (src_out.get_node().get_friendly_name(), src_out.get_index(), explicit_sign)
            if key in rotated_outputs:
                return rotated_outputs[key]
            x = src_out
            if explicit_sign:
                s = ops.constant(signs[k].reshape(1, 1, k).astype(np.float32))
                x = ops.multiply(x, s)
                n_sign_mul += 1
            r = ops.reshape(x, ops.constant(np.array([0, 0, k // block, block], np.int64)), True)
            h = ops.matmul(r, H, False, False)
            out = ops.reshape(h, ops.constant(np.array([0, 0, k], np.int64)), True)
            out.set_friendly_name(src_out.get_node().get_friendly_name() + "/hadamard")
            rotated_outputs[key] = out.output(0)
            n_rot += 1
            return out.output(0)

        for hf, mm in sorted(matmuls.items()):
            if hf == "lm_head":
                gname, layer, leaf = "output.weight", -1, "lm_head"
            else:
                lm = layer_re.match(hf)
                layer, leaf = int(lm.group(1)), lm.group(2)
                gname = f"blk.{layer}.{QUANT_MAP[leaf]}.weight"
            if gname not in folded:
                continue
            k = k_of[hf]
            if k not in signs:
                sys.exit(f"{hf}: folded with K={k} but no sign vector")
            src = mm.input_value(0)
            producer = src.get_node().get_friendly_name()
            # sign handling per input kind
            if leaf in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                        "mlp.gate_proj", "mlp.up_proj", "lm_head"):
                explicit = not fold_norm   # folded into the RMSNorm weight below
            elif leaf == "mlp.down_proj":
                explicit = not fold_up     # folded into up_proj rows
            else:
                explicit = True            # attention/GDN out_proj
            mm.input(0).replace_source_output(rotate(src, k, explicit))

        # RMSNorm sign folds (input_layernorm, post_attention_layernorm, final norm)
        for (layer, which), (node, c) in norm_nodes.items():
            if not fold_norm or which not in ("input_layernorm", "post_attention_layernorm", "norm"):
                continue
            k = int(np.prod(c.get_output_shape(0)))
            if k not in signs:
                continue
            consumers = [t.get_node() for t in node.output(0).get_target_inputs()]
            kinds = {cn.get_type_name() for cn in consumers}
            if not kinds <= {"MatMul", "ShapeOf", "Reshape"}:
                sys.exit(f"{node.get_friendly_name()}: unexpected consumers {kinds}")
            data = const_f32(c).reshape(-1) * signs[k]
            replace_const(c, data.reshape(c.get_output_shape(0)))
            n_norm_fold += 1
        print(f"[ir] hadamard: {n_rot} rotations inserted, {n_sign_mul} explicit sign multiplies, "
              f"{n_norm_fold} norm folds")

    # ---- embedding ------------------------------------------------------------
    t = tensors["token_embd.weight"]
    if t.tensor_type.name != "PQ2_0":
        sys.exit(f"token_embd is {t.tensor_type.name}, expected PQ2_0")
    if a.verify:
        emb = core.read_model(a.template_embed)
        emb_c = next(n for n in emb.get_ordered_ops()
                     if n.get_type_name() == "Constant" and len(n.get_output_shape(0)) == 2
                     and n.get_output_shape(0)[1] == int(g("embedding_length")))
        E = blocks_dequant(pq2_blocks(t))
        if folded:
            k = E.shape[1]
            Hn = hadamard(block).astype(np.float32)
            for b in range(k // block):
                E[:, b * block:(b + 1) * block] = E[:, b * block:(b + 1) * block] @ Hn
            E *= signs[k][None, :]
        rep.add("embedding", "token_embd", E, const_f32(emb_c))
        rep.dump()
        print("[ir] verify OK")
        return
    blocks = pq2_blocks(t)
    emb = ternary_embedding(blocks, block, signs[blocks.shape[1] * GROUP] if folded else None)
    print(f"[ir] embedding: ternary {blocks.shape[0]}x{blocks.shape[1] * GROUP}"
          f"{', inverse-rotated on the device' if folded else ''} ({time.perf_counter() - t0:.0f}s)")
    del blocks

    os.makedirs(a.out_dir, exist_ok=True)
    ov.save_model(model, os.path.join(a.out_dir, "openvino_model.xml"), compress_to_fp16=False)
    ov.save_model(emb, os.path.join(a.out_dir, "openvino_text_embeddings_model.xml"),
                  compress_to_fp16=False)
    print(f"[ir] wrote {a.out_dir} ({time.perf_counter() - t0:.0f}s)")


def _norm_folded_input(mm, folded, tensors, layer) -> bool:
    """True when the RMSNorm feeding this dense MatMul had its weight
    sign-folded (i.e. some sibling consumer is a folded projection)."""
    src = mm.input_value(0).get_node()
    for tgt in src.output(0).get_target_inputs():
        sib = tgt.get_node()
        m = re.match(re.escape(PREFIX) + re.escape(LM) + r"layers\.(\d+)\.(.+)/ov_ext::linear/MatMul$",
                     sib.get_friendly_name())
        if m and m.group(2) in QUANT_MAP:
            if f"blk.{m.group(1)}.{QUANT_MAP[m.group(2)]}.weight" in folded:
                return True
    return False


if __name__ == "__main__":
    main()
