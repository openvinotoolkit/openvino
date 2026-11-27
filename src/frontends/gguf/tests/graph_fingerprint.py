# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compact GGUF structural references; all graph capture lives in the C++ test runner."""

import argparse
import base64
import bisect
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DATA = HERE / "test_data" / "graph_regressions"
MAX_REFERENCE_BYTES = 32 * 1024
VERSION = 1
REFERENCE_VERSION = 3
RECIPE_VERSION = 2
SCRIPT_PATH = "src/frontends/gguf/tests/graph_fingerprint.py"
REFERENCES_PATH = "src/frontends/gguf/tests/test_data/graph_regressions/references"
README_PATH = "src/frontends/gguf/tests/graph_regressions/README.md"
POLICY_ANCHOR = "#review-policy-for-reference-changes"

# GGML block storage, not the frontend's unpacked representation.
BLOCKS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
    7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110),
    12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    30: (1, 2), 39: (32, 17), 41: (128, 18), 42: (64, 18),
    142: (128, 34), 143: (128, 28),
}
STORAGE_TYPES = {0: "f32", 1: "f16", 24: "i8", 25: "i16", 26: "i32",
                 27: "i64", 28: "f64", 30: "bf16"}
METADATA_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                    6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
GGML_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 24: "I8", 25: "I16",
              26: "I32", 27: "I64", 28: "F64", 30: "BF16", 39: "MXFP4", 41: "Q1_0", 42: "Q2_0",
              142: "PQ2_0", 143: "PTQ1_0"}
GGML_IDS = {name: kind for kind, name in GGML_NAMES.items()}
METADATA_NAMES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool",
                  8: "str", 10: "u64", 11: "i64", 12: "f64"}
METADATA_IDS = {name: kind for kind, name in METADATA_NAMES.items()}
LAYER_INDEX = re.compile(r"(?<![^.])\d+(?![^.])")
MIN_ENCODED_ARRAY = 64
OMIT_METADATA = {
    "tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges", "tokenizer.chat_template",
    "general.name", "general.description", "general.author", "general.license",
    "general.url",
}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def compact_snapshot(snapshot):
    result = {key: snapshot[key] for key in ("totals", "ops", "projections")}
    if not snapshot["canonical"].get("nodes") or not result["ops"]:
        raise ValueError("Snapshot must contain a nonempty graph and operation histogram")
    if sum(row[-1] for row in result["ops"]) != result["totals"]["nodes"]:
        raise ValueError("Operation histogram does not match captured node count")
    result["graph_sha256"] = digest(snapshot["canonical"])
    return result


def histogram_delta(expected, actual):
    before = {tuple(row[:-1]): row[-1] for row in expected}
    after = {tuple(row[:-1]): row[-1] for row in actual}
    return [{"signature": list(key), "delta": after.get(key, 0) - before.get(key, 0)}
            for key in sorted(before.keys() | after.keys())
            if before.get(key, 0) != after.get(key, 0)]


def model_delta(old, new):
    if old == new:
        return None
    return {
        "status": "added" if old is None else "removed" if new is None else "changed",
        "ops": histogram_delta(old["ops"] if old else [], new["ops"] if new else []),
        "projections": histogram_delta(old["projections"] if old else [], new["projections"] if new else []),
        "before_nodes": old["totals"]["nodes"] if old else 0,
        "after_nodes": new["totals"]["nodes"] if new else 0,
        "graph_changed": old is None or new is None or old["graph_sha256"] != new["graph_sha256"],
    }


def reference_delta(expected, actual):
    old = expected["models"] if expected else {}
    new = actual["models"]
    changes = {name: delta for name in sorted(old.keys() | new.keys())
               if (delta := model_delta(old.get(name), new.get(name)))}
    projectors_changed = expected is None or expected["projectors"] != actual["projectors"]
    return {"models": changes, "projectors_changed": projectors_changed} if changes or projectors_changed else None


def print_delta(case, name, delta):
    print(f"{case} [{name}]: {delta['status']}; operations {delta['before_nodes']} -> {delta['after_nodes']}")
    for field in ("ops", "projections"):
        for row in delta[field]:
            kind = "added" if row["delta"] > 0 else "removed"
            print(f"  {kind} {field}: {'/'.join(map(str, row['signature']))} {row['delta']:+d}")
    if delta["graph_changed"] and not delta["ops"]:
        print("  structure/attributes changed; operation counts unchanged")


def print_changes(case, change):
    for name, delta in change["models"].items():
        print_delta(case, name, delta)
    if change["projectors_changed"]:
        print(f"{case}: projector identities changed")


PROJECTORS = "projectors"


def change_signature(name, delta):
    if delta is None:
        return (PROJECTORS,)
    # Delta magnitudes scale with depth, so only the direction is part of the signature.
    rows = {field: tuple((tuple(row["signature"]), row["delta"] > 0) for row in delta[field])
            for field in ("ops", "projections")}
    # Without count deltas the graph name is the only discriminator left.
    scope = name if not rows["ops"] and not rows["projections"] else None
    return (scope, delta["status"], rows["ops"], rows["projections"], delta["graph_changed"])


def group_changes(changes):
    # Each named graph (adaptation variant) is grouped on its own, independent of its case's other graphs.
    groups = {}
    for change in changes:
        entries = list(change["change"]["models"].items())
        if change["change"]["projectors_changed"]:
            entries.append((PROJECTORS, None))
        for name, delta in entries:
            member = {"case": change["case"], "graph": name, "architecture": change["architecture"],
                      "models": change["models"], "delta": delta}
            groups.setdefault(change_signature(name, delta), []).append(member)
    result = []
    for members in sorted(groups.values(), key=lambda members: -len(members)):
        synthetic = ("fixture--", "representative--")
        representative = next((member for member in members if not member["case"].startswith(synthetic)),
                              members[0])
        result.append({
            "representative": {"case": representative["case"], "graph": representative["graph"]},
            "architectures": sorted({member["architecture"] for member in members}),
            "members": [{"case": member["case"], "graph": member["graph"]} for member in members],
            "cases": list(dict.fromkeys(member["case"] for member in members)),
            "models": sorted({model for member in members for model in member["models"]}),
            "delta": representative["delta"]})
    return result


def member_label(member):
    return f"{member['case']} [{member['graph']}]"


def print_groups(groups, limit=10):
    for index, group in enumerate(groups, 1):
        members = group["members"]
        print(f"\nChange group {index}: {len(members)} graph(s) in {len(group['cases'])} case(s), "
              f"{len(group['models'])} model alias(es); architectures: {', '.join(group['architectures'])}")
        representative = group["representative"]
        print(f"Representative (deltas shown for this graph): {member_label(representative)}")
        if group["delta"] is None:
            print(f"{representative['case']}: projector identities changed")
        else:
            print_delta(representative["case"], representative["graph"], group["delta"])
        listed = ", ".join(map(member_label, members[:limit]))
        if len(members) > limit:
            listed += f", ... and {len(members) - limit} more"
        print(f"  members: {listed}")


def record_groups(report):
    groups = group_changes(report["changed_cases"])
    report["change_groups"] = [{key: value for key, value in group.items() if key != "delta"} for group in groups]
    print_groups(groups)


def source_link(path, anchor=""):
    server, repository, sha = (os.environ.get(name) for name in
                               ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_SHA"))
    if server and repository and sha:
        return f"{server}/{repository}/blob/{sha}/{path}{anchor}"
    return f"{path}{anchor}"


RUNNER_HINT = ("Build the ov_gguf_frontend_tests target (-DENABLE_TESTS=ON) and point "
               "OV_GGUF_GRAPH_RUNNER at it. From the repository root, for the default "
               "Linux x86_64 Release output (with the src/frontends/gguf/docs/testing.md "
               "build use build-gguf/bin; adjust the path for other builds):")
RUNNER_EXPORT = 'export OV_GGUF_GRAPH_RUNNER="$PWD/bin/intel64/Release/ov_gguf_frontend_tests"'


def update_command(changes):
    ids = ",".join(change["case"] for change in changes)
    return f"python3 {SCRIPT_PATH} --cases {ids} --update"


def failure_guidance(changes, report_path):
    groups = group_changes(changes)
    return [
        f"GGUF structural regression check FAILED: {len(changes)} case(s) differ from the "
        f"committed references, in {len(groups)} group(s) by net change signature.",
        "Each named graph is grouped separately. A shared signature does not prove equivalent "
        "graph changes; inspect each group and select real checkpoints as described in the review policy:",
        *(f"  group {index}: {member_label(group['representative'])} ({len(group['members'])} graph(s) in "
          f"{len(group['cases'])} case(s); {', '.join(group['architectures'])})"
          for index, group in enumerate(groups, 1)),
        "If the change is unintended, fix the frontend change instead of the references.",
        "If the change is intended:",
        f"  1. {RUNNER_HINT}",
        f"     {RUNNER_EXPORT}",
        f"  2. Regenerate only these references with {SCRIPT_PATH}:",
        f"     {update_command(changes)}",
        f"  3. Commit the updated files in {REFERENCES_PATH}/.",
        "  4. Add real-model performance evidence for the affected models to the PR "
        "description, as required by the review policy:",
        f"     {source_link(README_PATH, POLICY_ANCHOR)}",
        f"Changed-case report (input for selecting representatives): {report_path}",
    ]


def publish_failure(changes, report_path):
    lines = failure_guidance(changes, report_path)
    print("\n" + "\n".join(lines), flush=True)
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    escaped = "%0A".join(line.replace("%", "%25").replace("\r", "%0D") for line in lines)
    print(f"::error title=GGUF structural references changed::{escaped}", flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    groups = group_changes(changes)
    rows = ["| Group | Graphs | Cases | Representative | Graph | Status | Operations | Changed operation types | "
            "Changed projections |", "|---|---|---|---|---|---|---|---|---|"]
    details = []
    for index, group in enumerate(groups, 1):
        representative, delta = group["representative"], group["delta"]
        prefix = (f"| {index} | {len(group['members'])} | {len(group['cases'])} | "
                  f"`{representative['case']}` | `{representative['graph']}` |")
        if delta is None:
            rows.append(f"{prefix} projector identities changed | - | - | - |")
        else:
            ops, projections = (", ".join(f"{'/'.join(map(str, row['signature']))} {row['delta']:+d}"
                                          for row in delta[field]) or "-" for field in ("ops", "projections"))
            if ops == "-" and projections == "-":
                ops = "structure/attributes only"
            rows.append(f"{prefix} {delta['status']} | {delta['before_nodes']} → {delta['after_nodes']} | "
                        f"{ops} | {projections} |")
        details += [f"<details><summary>Group {index}: {len(group['members'])} graph(s) in "
                    f"{len(group['cases'])} case(s); {', '.join(group['architectures'])}</summary>", "",
                    *(f"- `{member['case']}` `[{member['graph']}]`" for member in group["members"]),
                    "", "</details>", ""]
    text = "\n".join([
        "## GGUF structural regression check failed", "",
        f"{len(changes)} case(s) differ from the committed references, in {len(groups)} group(s) by "
        "net change signature. Each named graph is grouped separately. A shared signature does not "
        "prove equivalent graph changes; inspect each group and select real checkpoints according "
        "to the review policy. Operation deltas are shown for each group's representative.", "",
        *rows, "", *details,
        "If the change is unintended, fix the frontend change instead of the references. "
        "If it is intended:", "",
        f"1. {RUNNER_HINT}", "",
        "   ```sh", "   " + RUNNER_EXPORT, "   ```", "",
        f"2. Regenerate only these references with `{SCRIPT_PATH}`:", "",
        "   ```sh", "   " + update_command(changes), "   ```", "",
        f"3. Commit the updated files in `{REFERENCES_PATH}/`.",
        "4. Add real-model performance evidence for the affected models to the PR description "
        f"as required by the [review policy]({source_link(README_PATH, POLICY_ANCHOR)}).", "",
        "The changed-case report is uploaded in the `test-results-cpp` artifact as "
        "`changed_cases.json`.", ""])
    with open(summary, "a", encoding="utf-8") as stream:
        stream.write(text)


class HeaderReader:
    def __init__(self, stream):
        self.stream = stream

    def read(self, size):
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated GGUF header")
        return data

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        size = self.number("Q")
        if size > 64 * 1024 * 1024:
            raise ValueError("GGUF string exceeds header import limit")
        return self.read(size).decode("utf-8")

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind in METADATA_FORMATS:
            value = self.number(METADATA_FORMATS[kind])
            if isinstance(value, float) and not math.isfinite(value):
                return {"nonfinite": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
            return value
        raise ValueError(f"Unsupported GGUF metadata type {kind}")


def import_header(path, source_size=None):
    """source_size is the original file size when path holds only a header prefix."""
    return import_header_with_digest(path, source_size)[0]


def import_header_with_digest(path, source_size=None):
    omitted, metadata, tensors = [], [], []
    with Path(path).open("rb") as stream:
        reader = HeaderReader(stream)
        if reader.read(4) != b"GGUF" or reader.number("I") not in (2, 3):
            raise ValueError("Expected GGUF version 2 or 3")
        tensor_count, metadata_count = reader.number("Q"), reader.number("Q")
        if max(tensor_count, metadata_count) > 1000000:
            raise ValueError("GGUF header count exceeds import limit")
        for _ in range(metadata_count):
            name, kind = reader.string(), reader.number("I")
            entry = {"name": name, "type": kind}
            if kind == 9:
                element, count = reader.number("I"), reader.number("Q")
                if count > 10000000:
                    raise ValueError("GGUF array exceeds import limit")
                entry["element_type"] = element
                entry["value"] = [reader.value(element) for _ in range(count)]
            else:
                entry["value"] = reader.value(kind)
            if name in OMIT_METADATA:
                omitted.append(name)
            else:
                metadata.append(entry)
        for _ in range(tensor_count):
            name, rank = reader.string(), reader.number("I")
            if rank == 0 or rank > 4:
                raise ValueError(f"Invalid GGUF rank for {name}: {rank}")
            dimensions = [reader.number("Q") for _ in range(rank)]
            kind = reader.number("I")
            offset = reader.number("Q")
            if kind not in BLOCKS or any(d <= 0 for d in dimensions):
                raise ValueError(f"Unsupported tensor descriptor {name}: type={kind}, shape={dimensions}")
            block, size = BLOCKS[kind]
            if dimensions[0] % block:
                raise ValueError(f"Invalid block dimension for {name}")
            shape = list(reversed(dimensions))
            byte_count = math.prod(shape) // block * size
            if byte_count > (1 << 63) - 1:
                raise ValueError(f"Tensor byte count exceeds supported range: {name}")
            dtype = STORAGE_TYPES.get(kind, "u8")
            tensors.append({"name": name, "shape": shape, "ggml_type": kind,
                            "storage_type": dtype,
                            "storage_shape": shape if kind in STORAGE_TYPES else [byte_count],
                            "_offset": offset})
        header_size = stream.tell()
        stream.seek(0)
        header_sha256 = hashlib.sha256(reader.read(header_size)).hexdigest()
        alignment = next((entry["value"] for entry in metadata if entry["name"] == "general.alignment"), 32)
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Invalid GGUF alignment")
        data_start = (header_size + alignment - 1) // alignment * alignment
        offsets = sorted(tensor["_offset"] for tensor in tensors)
        file_size = Path(path).stat().st_size if source_size is None else source_size
        ambiguous, legacy = [], False
        for tensor in tensors:
            if tensor["ggml_type"] != 42:
                continue
            offset = tensor["_offset"]
            next_index = bisect.bisect_right(offsets, offset)
            span = (offsets[next_index] if next_index < len(offsets) else file_size - data_start) - offset
            candidates = []
            for kind in (42, 142):
                block, size = BLOCKS[kind]
                count = math.prod(tensor["shape"]) // block * size
                if tensor["shape"][-1] % block == 0 and count <= span and (
                        count == span or (count + alignment - 1) // alignment * alignment == span):
                    candidates.append(kind)
            if not candidates:
                raise ValueError(f"Invalid legacy Q2_0 packing for {tensor['name']}")
            if len(candidates) == 2:
                ambiguous.append(tensor)
            elif candidates == [142]:
                tensor["ggml_type"] = 142
                legacy = True
        if legacy:
            for tensor in ambiguous:
                tensor["ggml_type"] = 142
        for tensor in tensors:
            kind = tensor["ggml_type"]
            block, size = BLOCKS[kind]
            byte_count = math.prod(tensor["shape"]) // block * size
            if kind not in STORAGE_TYPES:
                tensor["storage_shape"] = [byte_count]
            del tensor["_offset"]
    recipe = {"version": VERSION, "metadata": metadata, "tensors": tensors,
              "omitted_metadata": sorted(omitted)}
    validate_recipe(recipe)
    return recipe, header_sha256


def unique(items, field):
    values = [item[field] for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {field} in input")


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or type(recipe.get("version")) is not int or recipe["version"] != VERSION:
        raise ValueError("Unsupported recipe version")
    if not isinstance(recipe.get("metadata"), list) or not recipe["metadata"] or (
            not isinstance(recipe.get("tensors"), list) or not recipe["tensors"]):
        raise ValueError("Recipe requires version 1, metadata and tensors")
    unique(recipe["metadata"], "name")
    unique(recipe["tensors"], "name")
    if not any(item["name"] == "general.architecture" for item in recipe["metadata"]):
        raise ValueError("Recipe has no general.architecture")
    integer_limits = {0: (0, 255), 1: (-128, 127), 2: (0, 65535), 3: (-32768, 32767),
                      4: (0, 2**32 - 1), 5: (-2**31, 2**31 - 1),
                      10: (0, 2**64 - 1), 11: (-2**63, 2**63 - 1)}
    for entry in recipe["metadata"]:
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise ValueError("Metadata names must be nonempty strings")
        kind = entry["type"]
        element = entry.get("element_type") if kind == 9 else kind
        if type(kind) is not int or type(element) is not int or element not in METADATA_FORMATS.keys() | {8}:
            raise ValueError(f"Invalid metadata type for {entry['name']}")
        values = entry["value"] if kind == 9 else [entry["value"]]
        if not isinstance(values, list):
            raise ValueError(f"Expected metadata array for {entry['name']}")
        for value in values:
            if element == 8:
                valid = type(value) is str
            elif element == 7:
                valid = type(value) is bool
            elif element in integer_limits:
                valid = type(value) is int and integer_limits[element][0] <= value <= integer_limits[element][1]
            else:
                maximum = 3.4028234663852886e38 if element == 6 else sys.float_info.max
                valid = ((type(value) in (float, int) and abs(value) <= maximum and math.isfinite(value)) or
                         (type(value) is dict and set(value) == {"nonfinite"} and
                          value["nonfinite"] in ("nan", "+inf", "-inf")))
            if not valid:
                raise ValueError(f"Invalid typed metadata value for {entry['name']}")
        if entry["name"] == "general.architecture" and (kind != 8 or not entry["value"]):
            raise ValueError("general.architecture must be a nonempty string scalar")
    for tensor in recipe["tensors"]:
        name, shape, kind = tensor["name"], tensor["shape"], tensor["ggml_type"]
        if not isinstance(name, str) or not name or not isinstance(shape, list) or not 1 <= len(shape) <= 4:
            raise ValueError("Invalid tensor name/rank")
        if type(kind) is not int or kind not in BLOCKS or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError(f"Invalid tensor descriptor: {name}")
        block, size = BLOCKS[kind]
        if shape[-1] % block:
            raise ValueError(f"Invalid block dimension for {name}")
        byte_count = math.prod(shape) // block * size
        if byte_count > (1 << 63) - 1:
            raise ValueError(f"Tensor byte count exceeds supported range: {name}")
        storage_shape = shape if kind in STORAGE_TYPES else [byte_count]
        if tensor["storage_type"] != STORAGE_TYPES.get(kind, "u8") or tensor["storage_shape"] != storage_shape:
            raise ValueError(f"Inconsistent source storage for {name}")


def storage(kind, shape):
    block, size = BLOCKS[kind]
    return STORAGE_TYPES.get(kind, "u8"), list(shape) if kind in STORAGE_TYPES else [math.prod(shape) // block * size]


def encode_array(element, values):
    if len(values) < MIN_ENCODED_ARRAY:
        return values
    if element in (1, 3, 5, 11) and all(type(value) is int and value in (-1, 1) for value in values):
        bits = bytearray((len(values) + 7) // 8)
        for index, value in enumerate(values):
            if value < 0:
                bits[index // 8] |= 1 << (index % 8)
        return {"signs": base64.b64encode(bytes(bits)).decode("ascii"), "count": len(values)}
    runs = []
    for value in values:
        if runs and canonical_bytes(runs[-1][0]) == canonical_bytes(value):
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return {"rle": runs} if len(runs) * 4 <= len(values) else values


def decode_array(name, value):
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"Invalid encoded metadata array for {name}")
    if set(value) == {"rle"} and isinstance(value["rle"], list):
        result = []
        for run in value["rle"]:
            if not isinstance(run, list) or len(run) != 2 or type(run[1]) is not int or run[1] < 1:
                raise ValueError(f"Invalid metadata run for {name}")
            result.extend(copy.deepcopy(run[0]) for _ in range(run[1]))
        return result
    if set(value) == {"signs", "count"} and type(value["count"]) is int and value["count"] >= 0:
        bits = base64.b64decode(value["signs"], validate=True)
        count = value["count"]
        if len(bits) != (count + 7) // 8 or (count % 8 and bits[-1] >> (count % 8)):
            raise ValueError(f"Invalid sign bits for {name}")
        return [-1 if bits[index // 8] >> (index % 8) & 1 else 1 for index in range(count)]
    raise ValueError(f"Invalid encoded metadata array for {name}")


def layer_ranges(layers):
    parts, start = [], layers[0]
    for index, layer in enumerate(layers):
        if index + 1 == len(layers) or layers[index + 1] != layer + 1:
            parts.append(str(start) if start == layer else f"{start}-{layer}")
            if index + 1 < len(layers):
                start = layers[index + 1]
    return ",".join(parts)


def parse_layer_ranges(name, text):
    layers = []
    for part in text.split(","):
        match = re.fullmatch(r"(0|[1-9]\d*)(?:-(0|[1-9]\d*))?", part)
        if not match:
            raise ValueError(f"Invalid layer range for {name}: {text}")
        first, last = int(match[1]), int(match[2] or match[1])
        if last < first or (layers and first <= layers[-1]):
            raise ValueError(f"Layer ranges must be ascending and disjoint for {name}: {text}")
        layers.extend(range(first, last + 1))
    return layers


def encode_recipe(recipe):
    """Compact on-disk form: typed metadata map and per-layer tensor templates without derived storage."""
    result = {"version": RECIPE_VERSION}
    for key in ("omitted_metadata",):
        if key in recipe:
            result[key] = recipe[key]
    result["metadata"] = {}
    for entry in recipe["metadata"]:
        if entry["type"] == 9:
            element = entry["element_type"]
            result["metadata"][entry["name"]] = [METADATA_NAMES[element] + "[]",
                                                 encode_array(element, entry["value"])]
        else:
            result["metadata"][entry["name"]] = [METADATA_NAMES[entry["type"]], entry["value"]]
    groups = {}
    for tensor in recipe["tensors"]:
        name = tensor["name"]
        if "{" in name or "}" in name:
            raise ValueError(f"Tensor name cannot be templated: {name}")
        match = LAYER_INDEX.search(name)
        if match and str(int(match[0])) == match[0]:
            key = (name[:match.start()] + "{}" + name[match.end():], tensor["ggml_type"], tuple(tensor["shape"]))
            groups.setdefault(key, []).append((int(match[0]), name))
        else:
            groups.setdefault((name, tensor["ggml_type"], tuple(tensor["shape"])), []).append((None, name))
    result["tensors"] = []
    for (template, kind, shape), members in groups.items():
        if len(members) == 1:
            result["tensors"].append([members[0][1], GGML_NAMES[kind], list(shape)])
        else:
            layers = sorted(layer for layer, _ in members)
            result["tensors"].append([template, GGML_NAMES[kind], list(shape), layer_ranges(layers)])
    return result


def decode_recipe(document):
    """Expand the on-disk recipe into the descriptor consumed by the C++ snapshot runner."""
    if not isinstance(document, dict) or document.get("version") != RECIPE_VERSION:
        raise ValueError("Unsupported recipe version")
    recipe = {"version": VERSION, "metadata": [], "tensors": []}
    for name, (kind, value) in document["metadata"].items():
        if kind.endswith("[]"):
            if kind[:-2] not in METADATA_IDS:
                raise ValueError(f"Invalid metadata type for {name}")
            recipe["metadata"].append({"name": name, "type": 9, "element_type": METADATA_IDS[kind[:-2]],
                                       "value": decode_array(name, value)})
        elif kind in METADATA_IDS:
            recipe["metadata"].append({"name": name, "type": METADATA_IDS[kind], "value": value})
        else:
            raise ValueError(f"Invalid metadata type for {name}")
    for entry in document["tensors"]:
        name, kind, shape = entry[:3]
        if kind not in GGML_IDS:
            raise ValueError(f"Unsupported tensor type for {name}: {kind}")
        if len(entry) == 4:
            if name.count("{}") != 1:
                raise ValueError(f"Layer template requires exactly one '{{}}': {name}")
            names = [name.replace("{}", str(layer)) for layer in parse_layer_ranges(name, entry[3])]
        elif "{" in name or "}" in name:
            raise ValueError(f"Tensor template requires layer ranges: {name}")
        else:
            names = [name]
        for concrete in names:
            storage_type, storage_shape = storage(GGML_IDS[kind], shape)
            recipe["tensors"].append({"name": concrete, "shape": list(shape), "ggml_type": GGML_IDS[kind],
                                      "storage_type": storage_type, "storage_shape": storage_shape})
    for key in ("omitted_metadata",):
        if key in document:
            recipe[key] = document[key]
    validate_recipe(recipe)
    return recipe


def recipe_text(recipe):
    """One metadata entry or tensor group per line keeps diffs reviewable."""
    document = encode_recipe(recipe)
    line = lambda value: json.dumps(value, allow_nan=False)
    lines = ["{", f'  "version": {document["version"]},']
    for key in ("omitted_metadata",):
        if key in document:
            lines.append(f"  {line(key)}: {line(document[key])},")
    lines.append('  "metadata": {')
    items = list(document["metadata"].items())
    lines += [f"    {line(name)}: {line(value)}{',' if index + 1 < len(items) else ''}"
              for index, (name, value) in enumerate(items)]
    lines += ["  },", '  "tensors": [']
    lines += [f"    {line(entry)}{',' if index + 1 < len(document['tensors']) else ''}"
              for index, entry in enumerate(document["tensors"])]
    lines += ["  ]", "}"]
    text = "\n".join(lines) + "\n"
    decoded = decode_recipe(json.loads(text))
    by_name = lambda value: sorted(value["tensors"], key=lambda tensor: tensor["name"])
    if (canonical_bytes({**decoded, "tensors": by_name(decoded)}) !=
            canonical_bytes({**recipe, "tensors": by_name(recipe)})):
        raise ValueError("Recipe encoding is not lossless")
    return text


def load_recipe(path):
    document = load_json(path)
    validate_document(document, "recipe")
    return decode_recipe(document)


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_document(value, kind):
    schema = load_json(DATA / "schema.json")

    def check(item, rule, location):
        if "$ref" in rule:
            target = schema
            for part in rule["$ref"].removeprefix("#/").split("/"):
                target = target[part]
            check(item, target, location)
            return
        if "oneOf" in rule:
            matches = 0
            for alternative in rule["oneOf"]:
                try:
                    check(item, alternative, location)
                    matches += 1
                except ValueError:
                    pass
            if matches != 1:
                raise ValueError(f"{location}: expected exactly one schema alternative")
        types = rule.get("type", [])
        types = [types] if isinstance(types, str) else types
        predicates = {
            "object": lambda: isinstance(item, dict),
            "array": lambda: isinstance(item, list),
            "string": lambda: isinstance(item, str),
            "integer": lambda: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda: isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item),
            "boolean": lambda: isinstance(item, bool),
        }
        if types and not any(predicates[name]() for name in types):
            raise ValueError(f"{location}: expected {types}")
        if "const" in rule and canonical_bytes(item) != canonical_bytes(rule["const"]):
            raise ValueError(f"{location}: unsupported value {item}")
        if "enum" in rule and canonical_bytes(item) not in [canonical_bytes(choice) for choice in rule["enum"]]:
            raise ValueError(f"{location}: unsupported value {item}")
        if isinstance(item, dict):
            missing = set(rule.get("required", [])) - item.keys()
            if missing or len(item) < rule.get("minProperties", 0):
                raise ValueError(f"{location}: missing required fields {sorted(missing)}")
            for key, child in item.items():
                if "propertyNames" in rule:
                    check(key, rule["propertyNames"], f"{location}.{key}")
                child_rule = rule.get("properties", {}).get(key, rule.get("additionalProperties", {}))
                if child_rule is False:
                    raise ValueError(f"{location}: unsupported semantic field {key}")
                if isinstance(child_rule, dict):
                    check(child, child_rule, f"{location}.{key}")
        if isinstance(item, list):
            if not rule.get("minItems", 0) <= len(item) <= rule.get("maxItems", sys.maxsize):
                raise ValueError(f"{location}: invalid array length")
            if rule.get("uniqueItems") and len({canonical_bytes(child) for child in item}) != len(item):
                raise ValueError(f"{location}: duplicate array values")
            for index, child in enumerate(item):
                prefix = rule.get("prefixItems", [])
                check(child, prefix[index] if index < len(prefix) else rule.get("items", {}),
                      f"{location}[{index}]")
        if isinstance(item, str):
            if len(item) < rule.get("minLength", 0) or (
                    "pattern" in rule and not re.search(rule["pattern"], item)):
                raise ValueError(f"{location}: invalid string")
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if not rule.get("minimum", -math.inf) <= item <= rule.get("maximum", math.inf):
                raise ValueError(f"{location}: numeric value out of range")

    check(value, schema["$defs"][kind], kind)


def contained_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Catalog path escapes data directory: {relative}")
    return path


def runner_path(explicit):
    path = explicit or os.environ.get("OV_GGUF_GRAPH_RUNNER") or shutil.which("ov_gguf_frontend_tests")
    if not path:
        raise ValueError("Set --runner or OV_GGUF_GRAPH_RUNNER to ov_gguf_frontend_tests")
    return path


def capture(recipe, runner):
    with tempfile.TemporaryDirectory(prefix="ov_gguf_graph_") as directory:
        input_path = Path(directory) / "recipe.json"
        input_path.write_text(json.dumps(recipe), encoding="utf-8")
        output = Path(directory) / "actual.json"
        command = [runner, "--gguf-snapshot", str(input_path), "--output", str(output)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
        except subprocess.TimeoutExpired as error:
            raise ValueError("Snapshot runner timed out after 120 seconds") from error
        if result.returncode:
            raise ValueError(f"Snapshot runner failed ({result.returncode}): {result.stderr[-4000:]}")
        snapshot = load_json(output)
        if not isinstance(snapshot.get("models"), dict) or not snapshot["models"]:
            raise ValueError("Snapshot runner must return a nonempty named model map")
        return {"models": {name: compact_snapshot(model) for name, model in snapshot["models"].items()},
                "projectors": snapshot["projectors"]}


def reference_bytes(reference):
    def format_value(value, level=0):
        indent = "  " * level
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = sorted(value.items())
            lines = ["{"]
            for index, (key, child) in enumerate(items):
                comma = "," if index + 1 < len(items) else ""
                lines.append(f'{"  " * (level + 1)}{json.dumps(key)}: {format_value(child, level + 1)}{comma}')
            lines.append(indent + "}")
            return "\n".join(lines)
        if isinstance(value, list):
            if not value:
                return "[]"
            if all(not isinstance(item, (dict, list)) for item in value):
                return json.dumps(value, separators=(",", ":"), allow_nan=False)
            if all(isinstance(item, list) and all(not isinstance(cell, (dict, list)) for cell in item)
                   for item in value):
                rows = [json.dumps(item, separators=(",", ":"), allow_nan=False) for item in value]
                return "[\n" + ",\n".join("  " * (level + 1) + row for row in rows) + "\n" + indent + "]"
            return "[\n" + ",\n".join("  " * (level + 1) + format_value(item, level + 1)
                                     for item in value) + "\n" + indent + "]"
        return json.dumps(value, allow_nan=False)

    data = (format_value(reference) + "\n").encode("utf-8")
    if len(data) > MAX_REFERENCE_BYTES:
        raise ValueError(f"Reference exceeds 32 KiB: {len(data)} bytes")
    return data


def install_references(pending):
    staged, committed = [], []
    preserved = set()
    try:
        for target, data in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            staged.append((None, target, None))
            if target.exists():
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    backup = Path(stream.name)
                    staged[-1] = (None, target, backup)
                    stream.write(target.read_bytes())
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                staged[-1] = (temporary, target, backup)
                stream.write(data)
        try:
            for temporary, target, backup in staged:
                os.replace(temporary, target)
                committed.append((target, backup))
        except OSError as error:
            for target, backup in reversed(committed):
                try:
                    if backup is None:
                        target.unlink()
                    else:
                        os.replace(backup, target)
                except OSError as rollback_error:
                    preserved.update(path for _, path in committed if path is not None and path.exists())
                    backups = ", ".join(f"{path} -> {original}" for original, path in committed if path in preserved)
                    new_targets = ", ".join(str(original) for original, path in committed
                                            if path is None and original.exists())
                    raise OSError(f"Publication failed ({error}); rollback failed for {target}: "
                                  f"{rollback_error}; retained backups: {backups}; "
                                  f"new references left published: {new_targets}") from rollback_error
            raise
    finally:
        for temporary, _, backup in staged:
            for path in (temporary, backup):
                if path is not None and path not in preserved and path.exists():
                    path.unlink()


def run_cases(args):
    report = {"selected": 0, "completed": 0, "changed_cases": [], "errors": []}
    report_path = args.report or Path.cwd() / "changed_cases.json"
    try:
        status = _run_cases(args, report)
        if status:
            publish_failure(report["changed_cases"], report_path)
        return status
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(str(error))
        if "change_groups" not in report:
            record_groups(report)
        raise
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"changed-case report: {report_path}")


def _run_cases(args, report):
    root = args.data.resolve()
    catalog = load_json(root / "catalog.json")
    validate_document(catalog, "catalog")
    unique(catalog["cases"], "id")
    unique(catalog["cases"], "reference")
    for case in catalog["cases"]:
        decoder = case["architecture"] != "clip" and "native_only" not in case
        if decoder != ("input_modes" in case):
            raise ValueError(f"{case['id']}: input_modes must be set only for adapted decoders")
        for alternate in case.get("equivalent_sources", []):
            if not set(alternate["models"]).issubset(case["models"]):
                raise ValueError(f"{case['id']}: equivalent source models must be listed in models")
    declared_recipes = {contained_path(root, case["recipe"]) for case in catalog["cases"]}
    declared_references = {contained_path(root, case["reference"]) for case in catalog["cases"]}
    if len(declared_references) != len(catalog["cases"]):
        raise ValueError("Duplicate reference target after resolving catalog paths")
    for directory, declared in (("recipes", declared_recipes), ("references", declared_references)):
        orphans = set((root / directory).glob("*.json")) - declared
        if orphans:
            raise ValueError(f"Orphan {directory}: {sorted(str(path.relative_to(root)) for path in orphans)}")
    if args.cases is not None and any(not name.strip() for name in args.cases.split(",")):
        raise ValueError("Unknown/empty case selection")
    requested = {name.strip() for name in args.cases.split(",")} if args.cases else {
        case["id"] for case in catalog["cases"]}
    known = {case["id"] for case in catalog["cases"]}
    if not requested or requested - known:
        raise ValueError(f"Unknown/empty case selection: {sorted(requested - known)}")
    if args.update and not args.cases:
        raise ValueError("--update requires explicit --cases")
    report["selected"] = len(requested)
    runner = runner_path(args.runner)
    changes, pending = report["changed_cases"], []
    for case in catalog["cases"]:
        if case["id"] not in requested:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"]):
            raise ValueError(f"Invalid case ID: {case['id']}")
        recipe_path = contained_path(root, case["recipe"])
        recipe = load_recipe(recipe_path)
        if "native_only" in case:
            recipe["native_only"] = True
        elif "input_modes" in case:
            recipe["input_modes"] = case["input_modes"]
        model = capture(recipe, runner)
        if "projectors" in case and model["projectors"] != case["projectors"]:
            raise ValueError(f"Projector identity mismatch for {case['id']}: {model['projectors']}")
        actual = {"version": REFERENCE_VERSION, **model}
        validate_document(actual, "reference")
        target = contained_path(root, case["reference"])
        if not target.exists() and not args.update:
            raise ValueError(f"Missing reference for {case['id']}; explicit --update required")
        expected = load_json(target) if target.exists() else None
        if args.update and expected is not None and expected.get("version") != REFERENCE_VERSION:
            expected = None
        if expected is not None:
            validate_document(expected, "reference")
            for name, component in expected["models"].items():
                if sum(row[-1] for row in component["ops"]) != component["totals"]["nodes"]:
                    raise ValueError(f"Reference histogram is inconsistent for {case['id']} [{name}]")
            reference_bytes(expected)
        delta = reference_delta(expected, actual)
        if delta:
            changes.append({"case": case["id"], "architecture": case["architecture"], "models": case["models"],
                            "change": delta})
        pending.append((target, reference_bytes(actual)))
        report["changed_cases"] = changes
        report["completed"] = len(pending)
    record_groups(report)
    if args.update:
        install_references(pending)
        for target, _ in pending:
            print(f"updated {target.relative_to(root)}")
    print(f"selected={len(pending)} changed={len(changes)} reference_bytes={sum(len(data) for _, data in pending)}")
    return 0 if args.update or not changes else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--cases")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--import-gguf", type=Path)
    parser.add_argument("--recipe-output", type=Path)
    parser.add_argument("--source-size", type=int, help="original file size when importing a header prefix")
    args = parser.parse_args(argv)
    try:
        if args.import_gguf:
            if not args.recipe_output or args.update or args.cases:
                raise ValueError("--import-gguf requires --recipe-output and cannot update references")
            recipe, header_sha256 = import_header_with_digest(args.import_gguf, args.source_size)
            args.recipe_output.write_text(recipe_text(recipe), encoding="utf-8")
            print(f"imported {len(recipe['tensors'])} tensors; omitted {recipe['omitted_metadata']}")
            print(f"catalog source.header_sha256: {header_sha256}")
            return 0
        return run_cases(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"GGUF graph reference error: {error}", file=sys.stderr)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            message = str(error).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error title=GGUF graph reference error::{message}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
