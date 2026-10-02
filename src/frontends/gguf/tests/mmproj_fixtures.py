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
    """Encoder inputs matching clip.cpp's Muse Glimmer window and shuffle ordering. Every window is
    padded to window^2 slots that repeat its first patch and are masked out as keys."""
    if min(height, width, window, merge) <= 0 or height % merge or width % merge:
        raise ValueError("Muse Glimmer grid must be positive and divisible by its merge size")
    groups = [[y * width + x
               for y in range(wy, min(wy + window, height))
               for x in range(wx, min(wx + window, width))]
              for wy in range(0, height, window) for wx in range(0, width, window)]
    slots = window * window
    order = np.array([i for group in groups for i in group + group[:1] * (slots - len(group))], np.int32)
    valid = np.array([k < len(group) for group in groups for k in range(slots)])
    inverse = np.empty(height * width, np.int32)
    inverse[order[valid]] = np.flatnonzero(valid)
    rows, cols = np.divmod(order, width)
    def indices(values):
        return np.asarray(values, np.int32).reshape(1, 1, 1, -1)
    return {"patch_indices": indices(order), "output_indices": indices(inverse),
            "position_x": indices(cols + 1), "position_y": indices(rows + 1),
            "merge_indices": indices(merge_window_order(height, width, merge)),
            "window_mask": np.where(valid, 0, -np.inf).astype(np.float32).reshape(len(groups), 1, 1, slots)}


WHISPER_AUDIO = {"qwen2a", "ultravox", "voxtral", "musicflamingo", "meralion", "glma"}
FIXED_VISION = {"gemma3", "idefics3", "internvl", "janus_pro", "mlp"}
DYNAMIC_VISION = {"gemma4v", "gemma4uv", "pixtral", "qwen2vl_merger", "qwen2.5vl_merger", "qwen3vl_merger",
                  "muse-glimmer"}


def vision_patch(metadata):
    """Pixels per encoder patch; unified Gemma4 patches span scale_factor reference patches."""
    patch = int(metadata["clip.vision.patch_size"])
    if metadata["vision.projector"] == "gemma4uv":
        patch *= int(metadata.get("clip.vision.projector.scale_factor", 3))
    return patch


def grid_unit(metadata, modality):
    """Image side granularity (pixels) or audio frame granularity a checkpoint accepts."""
    if modality == "audio":
        return 1 if metadata["audio.projector"] in {"gemma4a", "gemma4ua"} else 32
    return vision_patch(metadata) * int(metadata.get("vision.merge", 1))


def checkpoint_inputs(metadata, modality, width, height=None, seed=42):
    """Encoder inputs for a real checkpoint, read from its gguf_mmproj rt_info.

    Returns (feeds, raw, width, height): feeds name the converted model's inputs, and raw with
    width/height is what mmproj_oracle takes. For audio, width is the frame count.
    """
    projector = metadata[f"{modality}.projector"]
    rng = np.random.default_rng(seed)
    if modality == "audio":
        if projector == "gemma4ua":
            raw = rng.normal(0, .4, (640, width)).astype(np.float32)
            return {"audio.waveform_frames": raw.T.reshape(1, 1, width, 640)}, raw, width, 640
        mel = int(metadata["clip.audio.num_mel_bins"])
        raw = rng.normal(0, .4, (mel, width)).astype(np.float32)
        if projector == "gemma4a":
            return {"audio.features": raw.reshape(1, 1, mel, width)}, raw, width, mel
        if projector not in WHISPER_AUDIO:
            raise ValueError(f"No audio input contract for {projector}")
        ids = np.arange((width + 1) // 2, dtype=np.int32).reshape(1, 1, 1, -1)
        return {"audio.features": raw.reshape(1, mel, 1, width), "audio.position_ids": ids}, raw, width, mel
    if projector in FIXED_VISION:
        width = height = int(metadata["clip.vision.image_size"])
    elif projector not in DYNAMIC_VISION:
        raise ValueError(f"No vision input contract for {projector}")
    pixels = rng.uniform(0 if projector == "gemma4v" else -1, 1, (1, 3, height, width)).astype(np.float32)
    feeds = {"vision.pixel_values": pixels}
    raw = pixels[0].transpose(1, 2, 0)
    if projector in FIXED_VISION:
        return feeds, raw, width, height
    patch = vision_patch(metadata)
    gh, gw = height // patch, width // patch
    index = lambda values: np.asarray(values, np.int32).reshape(1, 1, 1, -1)
    if projector in {"gemma4v", "gemma4uv", "pixtral"}:
        rows, cols = np.indices((gh, gw))
        feeds.update({"vision.position_x": index(cols), "vision.position_y": index(rows)})
    elif projector == "muse-glimmer":
        feeds.update({"vision." + name: value for name, value in muse_glimmer_indices(
            gh, gw, int(metadata["vision.window_size"]), int(metadata["vision.merge"])).items()})
    else:
        # Qwen: a still image is a repeated temporal pair, grouped into 2x2 merge windows.
        order = np.array(merge_window_order(gh, gw), np.int32)
        if projector == "qwen2.5vl_merger":
            # llama.cpp's window attention: windows of window_size pixels in merged units.
            window = int(metadata.get("clip.vision.window_size", 112)) // patch // 2
            mh, mw = gh // 2, gw // 2
            groups, sizes = [], []
            for y in range(0, mh, window):
                for x in range(0, mw, window):
                    block = [yy * mw + xx for yy in range(y, min(y + window, mh)) for xx in range(x, min(x + window, mw))]
                    groups += block
                    sizes.append(4 * len(block))
            order = order.reshape(-1, 4)[groups].reshape(-1)
            mask = np.full((order.size, order.size), np.finfo(np.float32).min, np.float32)
            start = 0
            for size in sizes:
                mask[start:start + size, start:start + size] = 0
                start += size
            feeds["vision.attention_mask"] = mask[None, None]
            feeds["vision.output_indices"] = index(np.argsort(groups))
        rows, cols = np.divmod(order, gw)
        feeds["vision.pixel_values"] = np.repeat(pixels, 2, axis=0)
        feeds["vision.patch_indices"] = index(order)
        feeds["vision.position_ids"] = index(np.concatenate([rows, cols, rows, cols]))
    return feeds, raw, width, height
