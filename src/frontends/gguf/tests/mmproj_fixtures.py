# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the mmproj fixture generators."""
import io
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def finish(writer):
    """Flush a gguf.GGUFWriter and close it."""
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TensorWriter:
    """Random F32 fixture tensors added to a gguf.GGUFWriter; norm weights are centered at 1."""

    def __init__(self, writer, rng, std=.08):
        self.writer, self.rng, self.std = writer, rng, std

    def tensor(self, name, shape, norm=False):
        self.writer.add_tensor(name, (self.rng.normal(0, self.std, shape) + int(norm)).astype(np.float32))

    def linear(self, name, inp, out, bias=True):
        self.tensor(name + ".weight", (out, inp))
        if bias:
            self.tensor(name + ".bias", (out,))

    def norm(self, name, size, bias=True):
        self.tensor(name + ".weight", (size,), True)
        if bias:
            self.tensor(name + ".bias", (size,))


def run_oracle(oracle, model, modality, width, height, raw, env=None):
    """Run mmproj_oracle on `raw` and return its flat F32 embeddings."""
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        raw.tofile(directory / "input.f32")
        subprocess.run([str(Path(oracle).resolve()), str(model), modality, str(width), str(height),
                        str(directory / "input.f32"), str(directory / "output.f32")], check=True, env=env)
        return np.fromfile(directory / "output.f32", np.float32)


def merge_window_order(height, width, merge=2):
    """Grid indices with each merge x merge window contiguous, windows in row-major order."""
    return [(y + dy) * width + x + dx
            for y in range(0, height, merge) for x in range(0, width, merge)
            for dy in range(merge) for dx in range(merge)]


def split_variants(name, *suffixes):
    """Strip known fixture suffixes off `name`, returning (base, set of suffixes present)."""
    present = set()
    for suffix in suffixes:
        if name.endswith(suffix):
            present.add(suffix)
            name = name[: -len(suffix)]
    return name, present


def save_npz(path, arrays):
    """Write an .npz cnpy can read: NumPy's savez forces ZIP64 even for tiny arrays."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, array in arrays.items():
            payload = io.BytesIO()
            np.save(payload, np.ascontiguousarray(array))
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload.getvalue())


def muse_glimmer_indices(height, width, window, merge=2):
    """Encoder inputs matching clip.cpp's Muse Glimmer window and shuffle ordering."""
    if min(height, width, window, merge) <= 0 or height % merge or width % merge:
        raise ValueError("Muse Glimmer grid must be positive and divisible by its merge size")
    groups = [[y * width + x
               for y in range(wy, min(wy + window, height))
               for x in range(wx, min(wx + window, width))]
              for wy in range(0, height, window) for wx in range(0, width, window)]
    order = np.array([i for group in groups for i in group], np.int32)
    ids = np.repeat(np.arange(len(groups)), [len(group) for group in groups])
    rows, cols = np.divmod(order, width)
    def indices(values):
        return np.asarray(values, np.int32).reshape(1, 1, 1, -1)
    shuffle = merge_window_order(height, width, merge)
    return {"patch_indices": indices(order), "output_indices": indices(np.argsort(order)),
            "position_x": indices(cols + 1), "position_y": indices(rows + 1),
            "merge_indices": indices(shuffle),
            "attention_mask": np.where(ids[:, None] == ids[None, :], 0, -np.inf).astype(np.float32)[None, None]}
