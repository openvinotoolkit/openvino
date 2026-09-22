# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the mmproj fixture generators."""
import io
import zipfile

import numpy as np


def finish(writer):
    """Flush a gguf.GGUFWriter and close it."""
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


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
    shuffle = [(y + dy) * width + x + dx
               for y in range(0, height, merge) for x in range(0, width, merge)
               for dy in range(merge) for dx in range(merge)]
    return {"patch_indices": indices(order), "output_indices": indices(np.argsort(order)),
            "position_x": indices(cols + 1), "position_y": indices(rows + 1),
            "merge_indices": indices(shuffle),
            "attention_mask": np.where(ids[:, None] == ids[None, :], 0, -np.inf).astype(np.float32)[None, None]}
