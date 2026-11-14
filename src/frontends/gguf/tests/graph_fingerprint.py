# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compact GGUF structural references; all graph capture lives in the C++ test runner."""

import argparse
import bisect
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
OMIT_METADATA = {
    "tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges", "tokenizer.chat_template",
    "general.name", "general.description", "general.author", "general.license",
    "general.url", "general.tags",
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
    config = snapshot["canonical"].get("runtime", {}).get("gguf_mmproj", {})
    result["projectors"] = sorted(f"{key.removesuffix('.projector')}.{value}"
                                  for key, value in config.items() if key.endswith(".projector"))
    return result


def model_stage(models):
    if not models:
        raise ValueError("Snapshot stage requires a nonempty model map")
    return {"models": models, "map_sha256": digest({name: model["graph_sha256"] for name, model in models.items()})}


def histogram_delta(expected, actual):
    before = {tuple(row[:-1]): row[-1] for row in expected}
    after = {tuple(row[:-1]): row[-1] for row in actual}
    return [{"signature": list(key), "delta": after.get(key, 0) - before.get(key, 0)}
            for key in sorted(before.keys() | after.keys())
            if before.get(key, 0) != after.get(key, 0)]


def reference_delta(expected, actual):
    changes = []
    before_stages = expected.get("stages", {}) if expected else {}
    for stage in sorted(before_stages.keys() | actual["stages"].keys()):
        before = before_stages.get(stage, {}).get("models", {})
        after = actual["stages"].get(stage, {}).get("models", {})
        if before == after and before_stages.get(stage, {}).get("map_sha256") != actual["stages"].get(
                stage, {}).get("map_sha256"):
            changes.append({"stage": stage, "status": "map_changed"})
        for role in sorted(before.keys() | after.keys()):
            old, new = before.get(role), after.get(role)
            if old == new:
                continue
            changes.append({
                "stage": stage, "model": role,
                "status": "added" if old is None else "removed" if new is None else "changed",
                "ops": histogram_delta(old["ops"] if old else [], new["ops"] if new else []),
                "projections": histogram_delta(old["projections"] if old else [],
                                              new["projections"] if new else []),
                "before_nodes": old["totals"]["nodes"] if old else 0,
                "after_nodes": new["totals"]["nodes"] if new else 0,
                "graph_changed": old is None or new is None or old["graph_sha256"] != new["graph_sha256"],
            })
    if expected and expected.get("recipe_sha256") != actual["recipe_sha256"]:
        changes.append({"status": "recipe_changed"})
    return changes


def print_changes(case, changes):
    for change in changes:
        if change["status"] == "recipe_changed":
            print(f"{case}: recipe changed; review the input migration separately")
            continue
        if change["status"] == "map_changed":
            print(f"{case}: {change['stage']} model-map fingerprint changed")
            continue
        print(f"{case}: {change['stage']}/{change['model']} {change['status']}; "
              f"operations {change['before_nodes']} -> {change['after_nodes']}")
        for field in ("ops", "projections"):
            for row in change[field]:
                kind = "added" if row["delta"] > 0 else "removed"
                print(f"  {kind} {field}: {'/'.join(map(str, row['signature']))} {row['delta']:+d}")
        if change["graph_changed"] and not change["ops"]:
            print("  structure/attributes changed; operation counts unchanged")


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
              "omitted_metadata": sorted(omitted), "source_header_sha256": header_sha256}
    validate_recipe(recipe)
    return recipe


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


def materialize_recipe(recipe, path):
    """Bounded synthetic floats/packed-zero blocks, plus patterned Q8_0 controls."""
    payload = recipe.get("payload", {})
    if payload.get("mode") not in ("zeros", "pattern"):
        raise ValueError("Materialized recipe requires explicit zeros/pattern payload policy")
    supported = {0, 1} | (BLOCKS.keys() - STORAGE_TYPES.keys())
    if any(tensor["ggml_type"] not in supported for tensor in recipe["tensors"]):
        raise ValueError("Materialized generator requires F32, F16 or a supported packed format")
    if payload["mode"] != "zeros" and any(tensor["ggml_type"] not in (0, 1, 8)
                                         for tensor in recipe["tensors"]):
        raise ValueError("Packed formats other than Q8_0 require explicit zeros payload policy")
    total = sum(math.prod(tensor["storage_shape"]) *
                ({"f32": 4, "f16": 2, "u8": 1}.get(tensor["storage_type"], 0))
                for tensor in recipe["tensors"])
    if total > min(payload.get("max_bytes", 0), 64 * 1024 * 1024):
        raise ValueError(f"Synthetic payload exceeds the 64 MiB/per-recipe budget: {total}")

    def string(value):
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def typed(kind, value):
        if kind == 8:
            return string(value)
        if isinstance(value, dict):
            value = {"nan": math.nan, "+inf": math.inf, "-inf": -math.inf}[value["nonfinite"]]
        return struct.pack("<" + METADATA_FORMATS[kind], value)

    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(recipe["tensors"]), len(recipe["metadata"])))
    alignment = 32
    for entry in recipe["metadata"]:
        if entry["name"] == "general.alignment":
            alignment = entry["value"]
        header += string(entry["name"]) + struct.pack("<I", entry["type"])
        if entry["type"] == 9:
            header += struct.pack("<IQ", entry["element_type"], len(entry["value"]))
            for value in entry["value"]:
                header += typed(entry["element_type"], value)
        else:
            header += typed(entry["type"], entry["value"])
    if not isinstance(alignment, int) or alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError("Invalid synthetic GGUF alignment")
    offsets, offset = [], 0
    for tensor in recipe["tensors"]:
        offsets.append(offset)
        shape, kind = tensor["shape"], tensor["ggml_type"]
        header += string(tensor["name"]) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape)) + struct.pack("<IQ", kind, offset)
        block, size = BLOCKS[kind]
        offset += math.prod(shape) // block * size
        offset += (-offset) % alignment
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(bytes((-len(header)) % alignment))
        start = stream.tell()
        for tensor, offset in zip(recipe["tensors"], offsets):
            stream.write(bytes(start + offset - stream.tell()))
            count = math.prod(tensor["shape"])
            if tensor["ggml_type"] == 8:
                scale = payload.get("quant_scale", 0.0 if payload["mode"] == "zeros" else 0.125)
                block = struct.pack("<e", scale) + bytes(
                    0 if payload["mode"] == "zeros" else index % 16 for index in range(32))
                for _ in range(count // 32):
                    stream.write(block)
            elif tensor["ggml_type"] in (0, 1):
                code = "f" if tensor["ggml_type"] == 0 else "e"
                value = 0.0 if payload["mode"] == "zeros" else (
                    1.0 if "norm" in tensor["name"] else 0.125)
                chunk = struct.pack("<" + code, value) * min(count, 16384)
                while count:
                    take = min(count, 16384)
                    stream.write(chunk[:take * struct.calcsize(code)])
                    count -= take
            else:
                block, size = BLOCKS[tensor["ggml_type"]]
                remaining = count // block * size
                zeros = bytes(16384)
                while remaining:
                    take = min(remaining, len(zeros))
                    stream.write(zeros[:take])
                    remaining -= take


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


def capture(input_path, runner, materialized=False):
    with tempfile.TemporaryDirectory(prefix="ov_gguf_graph_") as directory:
        output = Path(directory) / "actual.json"
        command = [runner, "--gguf-snapshot", str(input_path), "--output", str(output)]
        if materialized:
            command.append("--materialized")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"Snapshot runner timed out after 120 seconds: {input_path}") from error
        if result.returncode:
            raise ValueError(f"Snapshot runner failed ({result.returncode}): {result.stderr[-4000:]}")
        return compact_snapshot(load_json(output))


def fingerprint(model_path, runner=None):
    """Explicit local-file compatibility mode; production materialization, no required-case skip."""
    result = capture(model_path, runner_path(runner), materialized=True)
    return {"sighash": result["graph_sha256"], "num_ops": result["totals"]["nodes"],
            "ops": result["ops"], "projections": result["projections"]}


def reference_bytes(reference):
    data = (json.dumps(reference, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
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
        return _run_cases(args, report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(str(error))
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
    changes, pending = [], []
    for case in catalog["cases"]:
        if case["id"] not in requested:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"]):
            raise ValueError(f"Invalid case ID: {case['id']}")
        recipe_path = contained_path(root, case["recipe"])
        recipe = load_json(recipe_path)
        validate_document(recipe, "recipe")
        validate_recipe(recipe)
        if recipe.get("source_header_sha256") and recipe["source_header_sha256"] != case["source"]["header_sha256"]:
            raise ValueError(f"Catalog/recipe header provenance differs for {case['id']}")
        stages = case.get("stages", [case.get("stage", "deferred_plan")])
        if case.get("kind", "model") == "model" and "deferred_plan" not in stages:
            raise ValueError(f"Model case {case['id']} requires deferred_plan")
        actual_stages = {}
        for stage in stages:
            if stage == "deferred_plan":
                model = capture(recipe_path, runner)
            elif stage in ("materialized", "controlled_values"):
                with tempfile.TemporaryDirectory(prefix="ov_gguf_materialized_") as directory:
                    path = Path(directory) / "synthetic.gguf"
                    materialize_recipe(recipe, path)
                    model = capture(path, runner, materialized=True)
            else:
                raise ValueError(f"Stage {stage} is unavailable until the adapted-map API is merged")
            actual_stages[stage] = model_stage({"native": model})
            if "projectors" in case and model["projectors"] != case["projectors"]:
                raise ValueError(f"Projector identity mismatch for {case['id']}: {model['projectors']}")
        actual = {"version": VERSION, "recipe_sha256": digest(recipe), "stages": actual_stages}
        validate_document(actual, "reference")
        target = contained_path(root, case["reference"])
        if not target.exists() and not args.update:
            raise ValueError(f"Missing reference for {case['id']}; explicit --update required")
        expected = load_json(target) if target.exists() else None
        if expected is not None:
            validate_document(expected, "reference")
            if set(expected["stages"]) != set(stages) and not args.update:
                raise ValueError(f"Reference has missing/extra required stages for {case['id']}")
            for stage in expected["stages"].values():
                if model_stage(stage["models"])["map_sha256"] != stage["map_sha256"]:
                    raise ValueError(f"Reference model-map hash is inconsistent for {case['id']}")
                for model in stage["models"].values():
                    if sum(row[-1] for row in model["ops"]) != model["totals"]["nodes"]:
                        raise ValueError(f"Reference histogram is inconsistent for {case['id']}")
            reference_bytes(expected)
        delta = reference_delta(expected, actual)
        if delta:
            print_changes(case["id"], delta)
            changes.append({"case": case["id"], "models": case["models"], "changes": delta})
        pending.append((target, reference_bytes(actual)))
        report["changed_cases"] = changes
        report["completed"] = len(pending)
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
    parser.add_argument("--materialized-file", type=Path,
                        help="Explicit local-model diagnostic, not the required recipe gate")
    args = parser.parse_args(argv)
    try:
        if args.materialized_file:
            if args.update or args.cases or args.import_gguf:
                raise ValueError("--materialized-file cannot select/update required cases")
            print(json.dumps(fingerprint(args.materialized_file, args.runner), indent=2))
            return 0
        if args.import_gguf:
            if not args.recipe_output or args.update or args.cases:
                raise ValueError("--import-gguf requires --recipe-output and cannot update references")
            recipe = import_header(args.import_gguf, args.source_size)
            args.recipe_output.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
            print(f"imported {len(recipe['tensors'])} tensors; omitted {recipe['omitted_metadata']}")
            return 0
        return run_cases(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"GGUF graph reference error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
