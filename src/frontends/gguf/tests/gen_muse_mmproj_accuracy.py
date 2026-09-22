# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer encoder oracle: llama.cpp 03fa73cb27f5c251b9528489b18d303b1366aca4.

Build mmproj_oracle.cpp against that revision's CPU-only libmtmd, then pass --oracle.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

import gguf
import numpy as np

from mmproj_fixtures import finish, muse_glimmer_indices, save_npz


def write_model(path):
    w = gguf.GGUFWriter(path, "clip")
    rng = np.random.default_rng(20260922)
    width, hidden, layers = 32, 48, 6
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "muse-glimmer")
    w.add_bool("clip.use_gelu", True)
    for name, value in {"embedding_length": width, "feed_forward_length": hidden,
                        "block_count": layers, "projection_dim": 12, "attention.head_count": 4,
                        "image_size": 4, "patch_size": 2, "spatial_merge_size": 2,
                        "image_min_pixels": 16, "image_max_pixels": 4096}.items():
        w.add_uint32("clip.vision." + name, value)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-5)
    w.add_array("clip.vision.image_mean", [.5, .5, .5])
    w.add_array("clip.vision.image_std", [.5, .5, .5])

    def tensor(name, shape, norm=False):
        w.add_tensor(name, (rng.normal(0, .08, shape) + int(norm)).astype(np.float32))

    def linear(name, inp, out, bias=True):
        tensor(name + ".weight", (out, inp))
        if bias:
            tensor(name + ".bias", (out,))

    def norm(name):
        tensor(name + ".weight", (width,), True)
        tensor(name + ".bias", (width,))

    tensor("v.patch_embd.weight", (width, 3, 2, 2))
    tensor("v.patch_embd.bias", (width,))
    tensor("v.position_embd.weight", (4, width))
    norm("v.pre_ln")
    norm("v.post_ln")
    for i in range(layers):
        p = f"v.blk.{i}."
        norm(p + "ln1")
        norm(p + "ln2")
        for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
            linear(p + name, width, width)
        linear(p + "ffn_up", width, hidden)
        linear(p + "ffn_down", hidden, width)
    linear("mm.0", width * 4, 24, False)
    linear("mm.1", 24, 20, False)
    linear("mm.2", 20, 12, False)
    finish(w)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data/mmproj_accuracy")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "model.gguf"
        write_model(model)
        combined = {}
        for step, (height, width) in enumerate([(8, 12), (4, 4)]):
            raw = np.random.default_rng(42).uniform(-1, 1, (height, width, 3)).astype(np.float32)
            raw.tofile(root / "input.f32")
            subprocess.run([str(args.oracle.resolve()), str(model), "vision", str(width), str(height),
                            str(root / "input.f32"), str(root / "output.f32")], check=True)
            arrays = dict(model=np.frombuffer(model.read_bytes(), np.uint8),
                          inputs=raw.transpose(2, 0, 1)[None],
                          embeddings=np.fromfile(root / "output.f32", np.float32).reshape(1, 1, -1, 12))
            arrays.update(muse_glimmer_indices(height // 2, width // 2, 2))
            if step == 0:
                combined.update(arrays)
            combined.update({f"{step}." + ("pixel_values" if name == "inputs" else name): value
                             for name, value in arrays.items() if name != "model"})
        save_npz(args.output / "muse-glimmer.npz", combined)


if __name__ == "__main__":
    main()
