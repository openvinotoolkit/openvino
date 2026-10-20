# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate DSpark full-logit fixtures using a CPU-only Prism llama.cpp oracle."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import gguf
import numpy as np

from mmproj_fixtures import finish, save_npz


def write_model(path):
    rng = np.random.default_rng(1234)
    w = gguf.GGUFWriter(path, "dspark")
    w.add_block_count(2)
    w.add_context_length(128)
    w.add_embedding_length(64)
    w.add_feed_forward_length(96)
    w.add_head_count(2)
    w.add_head_count_kv(1)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
    w.add_tokenizer_model("none")
    w.add_vocab_size(256)
    w.add_uint32("dspark.dspark.block_size", 4)
    w.add_uint32("dspark.dspark.mask_token_id", 255)
    w.add_array("dspark.dspark.target_layers", [1, 16])
    w.add_bool("dspark.dspark.log_snr_conditioning", True)
    w.add_float32("dspark.dspark.min_log_snr", -9.0)
    w.add_float32("dspark.dspark.max_log_snr", 9.0)

    def tensor(name, shape, norm=False):
        w.add_tensor(name, np.ones(shape, np.float32) if norm else rng.normal(0, 0.02, shape).astype(np.float32))

    tensor("token_embd.weight", (256, 64))
    tensor("output_norm.weight", (64,), True)
    tensor("dspark.fc.weight", (64, 128))
    tensor("dspark.hidden_norm.weight", (64,), True)
    for layer, width in ((1, 128), (2, 64)):
        tensor(f"dspark.log_snr_fc{layer}.weight", (64, width))
        tensor(f"dspark.log_snr_fc{layer}.bias", (64,))
    for layer in range(2):
        p = f"blk.{layer}."
        tensor(p + "attn_norm.weight", (64,), True)
        for kind, width in (("q", 64), ("k", 32), ("v", 32)):
            tensor(p + f"attn_{kind}.weight", (width, 64))
        tensor(p + "attn_output.weight", (64, 64))
        for kind in ("q", "k"):
            tensor(p + f"attn_{kind}_norm.weight", (32,), True)
        tensor(p + "ffn_norm.weight", (64,), True)
        tensor(p + "ffn_gate.weight", (96, 64))
        tensor(p + "ffn_down.weight", (64, 96))
        tensor(p + "ffn_up.weight", (96, 64))
    w.add_uint32("dspark.dspark.markov_rank", 8)
    w.add_bool("dspark.dspark.confidence_head", True)
    w.add_bool("dspark.dspark.confidence_head_with_markov", True)
    for name, shape in (
        ("dspark.markov_head_a.weight", (256, 8)),
        ("dspark.markov_head_b.weight", (256, 8)),
        ("dspark.confidence_head.weight", (1, 72)),
        ("dspark.confidence_head.bias", (1,)),
    ):
        tensor(name, shape)
    finish(w)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data/arch_accuracy/dspark.npz")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        model, reference = Path(directory) / "model.gguf", Path(directory) / "reference.bin"
        write_model(model)
        subprocess.run([args.oracle, str(model), str(reference)], check=True)
        with reference.open("rb") as f:
            width, block, vocab = np.frombuffer(f.read(12), np.int32)
            arrays = {"model": np.frombuffer(model.read_bytes(), np.uint8), "width": np.array([width])}
            reader = gguf.GGUFReader(model)
            for t in reader.tensors:
                if t.name.startswith(("dspark.markov", "dspark.confidence")):
                    arrays[t.name] = t.data.copy()
            for rows in (2, 3):
                assert np.frombuffer(f.read(4), np.int32)[0] == rows
                arrays[f"logits_{rows}"] = np.frombuffer(f.read(int(block * vocab * 4)), np.float32).reshape(
                    block, vocab
                )
            assert not f.read(1)
        save_npz(args.output, arrays)


if __name__ == "__main__":
    main()
