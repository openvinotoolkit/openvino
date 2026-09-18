# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate nonzero mmproj fixtures using llama.cpp CPU, without OpenVINO.

PYTHONPATH=<llama.cpp>/gguf-py python gen_mmproj_accuracy.py --oracle /path/to/mmproj_oracle
Reference revision: 16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np


def write_model(path, projector, projection_width=6):
    projector, _, variant = projector.partition("_") if projector.startswith("gemma3_") else (projector, "", "")
    audio = projector in {"qwen2a", "ultravox", "voxtral", "musicflamingo", "meralion", "glma"}
    clip = projector in {"mlp", "mlp_norm"}
    qwen = projector in {"qwen2vl_merger", "qwen2.5vl_merger", "qwen3vl_merger"}
    modality, prefix = ("audio", "a.") if audio else ("vision", "v.")
    key = f"clip.{modality}."
    w = gguf.GGUFWriter(path, "clip")
    w.add_bool(f"clip.has_{modality}_encoder", True)
    w.add_string("clip.projector_type", "mlp" if projector == "mlp_norm" else projector)
    w.add_bool("clip.use_gelu", True)
    for name, value in {"embedding_length": 8, "feed_forward_length": 12, "block_count": 2,
                        "projection_dim": projection_width, "attention.head_count": 2}.items():
        w.add_uint32(key + name, value)
    w.add_float32(key + "attention.layer_norm_epsilon", 1e-5)
    rng = np.random.default_rng(1729)

    def tensor(name, shape, norm=False):
        if projector == "internvl" and name.startswith("mm."):
            name = name.replace("mm.", "mm.model.mlp.", 1)
        if variant == "legacy":
            name = name.replace("ffn_up", "ffn_tmp").replace("ffn_down", "ffn_up").replace("ffn_tmp", "ffn_down")
        values = rng.normal(0, 0.08, shape).astype(np.float32)
        w.add_tensor(name, values + 1 if norm else values)

    tensor(prefix + "position_embd.weight", (17 if clip or projector == "internvl" else 16, 8))
    if clip or projector == "internvl":
        tensor("v.class_embd", (1, 8))
    for i in range(2):
        p = prefix + f"blk.{i}."
        for name in ("ln1", "ln2"):
            tensor(p + name + ".weight", (8,), True)
            tensor(p + name + ".bias", (8,))
        for name in (("attn_out",) if variant == "fused" else ("attn_q", "attn_k", "attn_v", "attn_out")):
            tensor(p + name + ".weight", (8, 8))
            if not (audio and name == "attn_k"):
                tensor(p + name + ".bias", (8,))
        if variant == "fused":
            tensor(p + "attn_qkv.weight", (24, 8))
            tensor(p + "attn_qkv.bias", (24,))
            tensor(p + "attn_q_norm.weight", (4,), True)
            tensor(p + "attn_k_norm.weight", (4,), True)
            tensor(p + "ls1.weight", (8,))
            tensor(p + "ls2.weight", (8,))
            tensor(p + "out_scale.weight", (8,), True)
            tensor(p + "attn_post_norm.weight", (8,), True)
            tensor(p + "ffn_post_norm.weight", (8,), True)
        tensor(p + "ffn_up.weight", (12, 8))
        tensor(p + "ffn_up.bias", (12,))
        tensor(p + "ffn_down.weight", (8, 12))
        tensor(p + "ffn_down.bias", (8,))
        if projector == "qwen3vl_merger":
            # The reference requires a fused projection for Qwen3 VL.
            tensor(p + "attn_qkv.weight", (24, 8))
            tensor(p + "attn_qkv.bias", (24,))
    if projector == "qwen3vl_merger":
        tensor("v.deepstack.0.norm.weight", (32,), True)
        tensor("v.deepstack.0.norm.bias", (32,))
        tensor("v.deepstack.0.fc1.weight", (12, 32))
        tensor("v.deepstack.0.fc1.bias", (12,))
        tensor("v.deepstack.0.fc2.weight", (6, 12))
        tensor("v.deepstack.0.fc2.bias", (6,))
    tensor(prefix + "post_ln.weight", (8,), True)
    tensor(prefix + "post_ln.bias", (8,))
    if not audio:
        w.add_uint32(key + "image_size", 8)
        w.add_uint32(key + "patch_size", 2)
        w.add_uint32(key + "projector.scale_factor", 2)
        w.add_array(key + "image_mean", [0.5, 0.5, 0.5])
        w.add_array(key + "image_std", [0.5, 0.5, 0.5])
        tensor("v.patch_embd.weight", (8, 3, 2, 2))
        if not qwen or projector == "qwen3vl_merger":
            tensor("v.patch_embd.bias", (8,))
        if qwen:
            w.add_uint32(key + "spatial_merge_size", 2)
            w.add_uint32(key + "n_wa_pattern", 2 if projector == "qwen2.5vl_merger" else 0)
            tensor("v.patch_embd.weight.1", (8, 3, 2, 2))
        if projector == "gemma3":
            tensor("mm.soft_emb_norm.weight", (8,), True)
            tensor("mm.input_projection.weight", (8, projection_width))
        elif projector == "idefics3":
            tensor("mm.model.fc.weight", (6, 32))
        elif qwen:
            tensor("mm.0.weight", (12, 32))
            tensor("mm.0.bias", (12,))
            tensor("mm.2.weight", (6, 12))
            tensor("mm.2.bias", (6,))
        elif projector == "internvl":
            tensor("mm.0.weight", (32,), True)
            tensor("mm.0.bias", (32,))
            tensor("mm.1.weight", (12, 32))
            tensor("mm.1.bias", (12,))
            tensor("mm.3.weight", (6, 12))
            tensor("mm.3.bias", (6,))
        elif clip:
            tensor("mm.0.weight", (12, 8))
            tensor("mm.0.bias", (12,))
            tensor("mm.3.weight" if projector == "mlp_norm" else "mm.2.weight", (6, 12))
            tensor("mm.3.bias" if projector == "mlp_norm" else "mm.2.bias", (6,))
            if projector == "mlp_norm":
                tensor("mm.1.weight", (12,), True)
                tensor("mm.1.bias", (12,))
                tensor("mm.4.weight", (6,), True)
                tensor("mm.4.bias", (6,))
        else:
            tensor("mm.0.weight", (12, 8))
            tensor("mm.0.bias", (12,))
            tensor("mm.1.weight", (6, 12))
            tensor("mm.1.bias", (6,))
    else:
        w.add_uint32(key + "num_mel_bins", 4)
        w.add_uint32(key + "projector.stack_factor", 2)
        tensor("a.conv1d.1.weight", (8, 4, 3))
        tensor("a.conv1d.1.bias", (8, 1))
        tensor("a.conv1d.2.weight", (8, 8, 3))
        tensor("a.conv1d.2.bias", (8, 1))
        if projector == "qwen2a":
            tensor("mm.a.fc.weight", (6, 8))
            tensor("mm.a.fc.bias", (6,))
        elif projector == "meralion":
            tensor("mm.a.norm_pre.weight", (16,), True)
            tensor("mm.a.norm_pre.bias", (16,))
            for i, shape in enumerate(((12, 16), (12, 12), (12, 12), (6, 12))):
                tensor(f"mm.a.mlp.{i}.weight", shape)
                tensor(f"mm.a.mlp.{i}.bias", (shape[0],))
        else:
            dim = 8 if projector == "musicflamingo" else 16
            tensor("mm.a.mlp.1.weight", (24 if projector == "ultravox" else 12, dim))
            tensor("mm.a.mlp.2.weight", (6, 12))
            if projector == "ultravox":
                tensor("mm.a.norm_pre.weight", (16,), True)
                tensor("mm.a.norm_mid.weight", (12,), True)
            if projector == "musicflamingo":
                tensor("mm.a.mlp.1.bias", (12,))
                tensor("mm.a.mlp.2.bias", (6,))
            if projector == "glma":
                tensor("mm.a.norm_pre.weight", (8,), True)
                tensor("mm.a.norm_pre.bias", (8,))
                tensor("mm.a.mlp.1.bias", (12,))
                tensor("mm.a.mlp.2.bias", (6,))
                tensor("v.boi", (1, 6))
                tensor("v.eoi", (1, 6))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data/mmproj_accuracy")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for projector in ("gemma3", "gemma3_fused", "gemma3_legacy", "idefics3", "janus_pro", "mlp", "mlp_norm",
                      "qwen2a", "ultravox", "voxtral", "musicflamingo", "meralion", "glma",
                      "qwen2vl_merger", "qwen2.5vl_merger", "qwen3vl_merger", "internvl",
                      "qwen2.5vl_merger_window_video", "voxtral_odd"):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "model.gguf"
            window_video = projector.endswith("_window_video")
            audio = write_model(path, projector.removesuffix("_window_video").removesuffix("_odd"))
            qwen = "vl_merger" in projector
            width, height = (120, 16) if window_video else (16, 4) if audio else (12, 8) if qwen else (8, 8)
            if projector.endswith("_odd"):
                width = 18
            shape = (height, width) if audio else (height, width, 3)
            raw = np.random.default_rng(42).normal(0, 0.4, shape).astype(np.float32)
            raw.tofile(directory / "input.bin")
            second = []
            if window_video:
                raw_second = np.random.default_rng(43).normal(0, 0.4, shape).astype(np.float32)
                raw_second.tofile(directory / "second.bin")
                second = [str(directory / "second.bin")]
            subprocess.run([str(args.oracle.resolve()), str(path), "audio" if audio else "vision",
                            str(width), str(height), str(directory / "input.bin"),
                            str(directory / "output.bin"), *second], check=True)
            output_width = 12 if projector == "qwen3vl_merger" else 6
            output = np.fromfile(directory / "output.bin", dtype=np.float32).reshape(1, 1, -1, output_width)
            inputs = raw.reshape(1, height, 1, width) if audio else raw.transpose(2, 0, 1)[None]
            extra = {}
            if qwen:
                inputs = np.repeat(inputs, 2, axis=0)
                indices = [y * (width // 2) + x + dy * (width // 2) + dx
                           for y in range(0, height // 2, 2) for x in range(0, width // 2, 2)
                           for dy in range(2) for dx in range(2)]
                rows, cols = np.divmod(indices, width // 2)
                extra["patch_indices"] = np.array(indices, np.int32).reshape(1, 1, 1, -1)
                extra["position_ids"] = np.array([rows, cols, rows, cols], np.int32).reshape(1, 1, 1, -1)
                extra["attention_mask"] = np.zeros((1, 1, len(indices), len(indices)), np.float32)
                extra["output_indices"] = np.arange(len(indices) // 4, dtype=np.int32).reshape(1, 1, 1, -1)
                if window_video:
                    inputs[1] = raw_second.transpose(2, 0, 1)
                    groups, windows = [], []
                    gh, gw = height // 4, width // 4
                    for y in range(0, gh, 28):
                        for x in range(0, gw, 28):
                            window = [yy * gw + xx for yy in range(y, min(y + 28, gh))
                                      for xx in range(x, min(x + 28, gw))]
                            groups.extend(window)
                            windows.append(len(window) * 4)
                    permutation = np.array([4 * group + i for group in groups for i in range(4)])
                    extra["patch_indices"] = extra["patch_indices"][..., permutation]
                    extra["position_ids"] = np.array([rows[permutation], cols[permutation]] * 2,
                                                       np.int32).reshape(1, 1, 1, -1)
                    mask = np.full((len(indices), len(indices)), np.finfo(np.float32).min, np.float32)
                    offset = 0
                    for count in windows:
                        mask[offset:offset + count, offset:offset + count] = 0
                        offset += count
                    extra["attention_mask"] = mask[None, None]
                    extra["output_indices"] = np.argsort(groups).astype(np.int32).reshape(1, 1, 1, -1)
            np.savez_compressed(args.output / f"{projector}.npz", model=np.frombuffer(path.read_bytes(), np.uint8),
                                inputs=inputs, embeddings=output, **extra)


if __name__ == "__main__":
    main()
