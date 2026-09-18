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
