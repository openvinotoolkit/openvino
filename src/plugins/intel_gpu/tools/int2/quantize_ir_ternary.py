"""Rewrite the fp16 MatMul weights of an exported LLM into ternary u2 weights.

OpenVINO has no signed 2-bit type, so ternary is expressed the way the GPU
plugin's weight-compression pattern expects: u2 codes {0,1,2} with a scalar zero
point of 1, times a per-group fp16 scale. Weights that are already exactly
ternary are reproduced rather than approximated.
"""

import argparse
import gc
import numpy as np
import openvino as ov
from openvino import opset13 as ops

GROUP = 128


def pack_u2(codes: np.ndarray) -> np.ndarray:
    """[..] uint8 in {0,1,2} -> packed u2, value j at bits [2j, 2j+1]."""
    flat = codes.reshape(-1).astype(np.uint8) & 0x3
    if flat.size % 4:
        flat = np.pad(flat, (0, 4 - flat.size % 4))
    q = flat.reshape(-1, 4)
    return (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8)


def to_f32(const) -> np.ndarray:
    """Constant data as fp32, decoding bf16 by hand.

    numpy has no bf16, so Constant.get_data() hands back a float16-typed *view*
    of the raw bf16 bits rather than a conversion. Reading that as fp16 keeps the
    ternary codes (the sign is bit 15 either way) but corrupts every group scale,
    which is silent: the model loads and generates fluent nonsense.
    """
    raw = const.get_data()
    if const.get_output_element_type(0) == ov.Type.bf16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32)


def ternary_quantize(w: np.ndarray):
    """w is [N, K] fp32. Returns u2 codes [N, K] and fp16 scales [N, K/GROUP]."""
    n, k = w.shape
    g = w.reshape(n, k // GROUP, GROUP)
    scale = np.abs(g).max(axis=2)
    safe = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint(g / safe[:, :, None]), -1, 1).astype(np.int8).reshape(n, k)
    return (q + 1).astype(np.uint8), scale.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--min-k", type=int, default=1024, help="skip tiny MatMuls")
    args = ap.parse_args()

    core = ov.Core()
    model = core.read_model(args.src)

    targets = []
    for node in model.get_ordered_ops():
        if node.get_type_name() != "MatMul":
            continue
        w = node.input_value(1).get_node()
        # optimum emits fp16 weights behind a Convert to the inference precision
        conv = None
        if w.get_type_name() == "Convert":
            conv, w = w, w.input_value(0).get_node()
        if w.get_type_name() != "Constant":
            continue
        shape = list(w.get_output_shape(0))
        if len(shape) != 2 or shape[1] % GROUP or shape[1] < args.min_k:
            continue
        targets.append((node, conv, w, shape))

    print(f"rewriting {len(targets)} MatMul weights")
    total_in = total_out = 0
    for i, (mm, conv, const, (n, k)) in enumerate(targets):
        # whatever fed the MatMul before, the replacement has to keep that type
        orig_et = mm.input_value(1).get_element_type()
        w = to_f32(const)
        codes, scale = ternary_quantize(w)
        total_in += w.size * 2
        del w

        packed = pack_u2(codes)
        t = ov.Tensor(ov.Type.u2, [n, k // GROUP, GROUP])
        np.frombuffer(t.data, dtype=np.uint8)[: packed.size] = packed
        wq = ops.constant(t)
        del codes, packed

        zp = ops.constant(np.array(1.0, np.float16).reshape(1, 1, 1))
        sc = ops.constant(scale.reshape(n, k // GROUP, 1))
        total_out += scale.size * 2 + (n * k) // 4
        del scale

        deq = ops.multiply(ops.subtract(ops.convert(wq, "f16"), zp), sc)
        flat = ops.reshape(deq, ops.constant(np.array([n, k], np.int64)), False)
        out = flat if orig_et == ov.Type.f16 else ops.convert(flat, orig_et)
        mm.input(1).replace_source_output(out.output(0))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(targets)}")
            gc.collect()

    print(f"weights {total_in / 2**30:.2f} GiB -> {total_out / 2**30:.2f} GiB")
    ov.save_model(model, args.dst, compress_to_fp16=False)
    print("saved", args.dst)


if __name__ == "__main__":
    main()
