# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Helper to count SDPA layers in an OpenVINO IR XML and build a positional
# KV_CACHE_PER_LAYER_CONFIG list for ov::Core::compile_model().
#
# Usage:
#   from per_layer_config_helper import (
#       count_sdpa_layers, make_uniform_config, make_layered_config,
#   )
#
#   # 1. Count SDPA layers in the model.
#   n = count_sdpa_layers("/path/to/openvino_model.xml")
#
#   # 2. Build a per-layer list. Example: first 4 TBQ-u4, rest u8 by-token.
#   cfg = make_layered_config(n, [
#       (4,  {"KEY_CACHE_QUANT_ALG":   "TURBO",
#             "VALUE_CACHE_QUANT_ALG": "TURBO",
#             "KEY_CACHE_PRECISION":   "u4",
#             "VALUE_CACHE_PRECISION": "u4"}),
#       (None, {"KEY_CACHE_QUANT_ALG":   "BY_TOKEN",
#               "VALUE_CACHE_QUANT_ALG": "BY_TOKEN",
#               "KEY_CACHE_PRECISION":   "u8",
#               "VALUE_CACHE_PRECISION": "u8",
#               "KEY_CACHE_GROUP_SIZE":  128,
#               "VALUE_CACHE_GROUP_SIZE": 128}),
#   ])
#
#   # 3. Pass to compile_model.
#   core.compile_model(model, "CPU", {"KV_CACHE_PER_LAYER_CONFIG": cfg})

import xml.etree.ElementTree as ET

SDPA_TYPES = {
    "ScaledDotProductAttention",       # ov::op::v13::ScaledDotProductAttention
    "ScaledDotProductAttentionWithKVCache",  # CPU plugin internal op
    "SDPAWithTransposeReshape",        # CPU plugin internal op
}


def count_sdpa_layers(xml_path):
    """Return the number of SDPA-like layers in an OpenVINO IR XML."""
    root = ET.parse(xml_path).getroot()
    return sum(1 for layer in root.iter("layer")
               if layer.get("type") in SDPA_TYPES)


def list_sdpa_layers(xml_path):
    """Return [(index, name, type), ...] for SDPA-like layers in topo order."""
    root = ET.parse(xml_path).getroot()
    out = []
    idx = 0
    for layer in root.iter("layer"):
        if layer.get("type") in SDPA_TYPES:
            out.append((idx, layer.get("name"), layer.get("type")))
            idx += 1
    return out


def make_uniform_config(n_layers, spec):
    """All layers share the same spec (subset of KV cache properties)."""
    return [dict(spec) for _ in range(n_layers)]


def make_layered_config(n_layers, ranges):
    """Build positional config from sequential (count, spec) ranges.

    `ranges` is a list of (count, spec) tuples; the last entry may use
    count=None to mean "fill the remainder". Ranges are applied in order.
    """
    out = []
    remaining = n_layers
    for i, (count, spec) in enumerate(ranges):
        if count is None:
            if i != len(ranges) - 1:
                raise ValueError("count=None only allowed in the last range")
            count = remaining
        if count > remaining:
            raise ValueError(
                f"range {i} count={count} exceeds remaining {remaining}")
        out.extend(dict(spec) for _ in range(count))
        remaining -= count
    if remaining != 0:
        raise ValueError(
            f"total range count {n_layers - remaining} != n_layers {n_layers}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    args = ap.parse_args()
    layers = list_sdpa_layers(args.xml)
    print(f"SDPA layers: {len(layers)}")
    for idx, name, op_type in layers:
        print(f"  [{idx:3d}] {op_type}  {name}")
