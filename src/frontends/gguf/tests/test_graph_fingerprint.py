# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Required recipe gate and focused standard-library tooling contracts."""

from argparse import Namespace
import copy
import json
from pathlib import Path
import struct

import pytest

import graph_fingerprint as graph


def recipe():
    return {
        "version": 1,
        "metadata": [{"name": "general.architecture", "type": 8, "value": "llama"}],
        "tensors": [{"name": "weight", "ggml_type": 0, "shape": [2, 2],
                     "storage_type": "f32", "storage_shape": [2, 2]}],
    }


def snapshot():
    return {"canonical": {"nodes": [{"type": "MatMul", "inputs": [0, 1]}]},
            "ops": [["MatMul", "opset1", 1]], "projections": [],
            "totals": {"nodes": 1, "top_level_nodes": 1, "edges": 2, "states": 0}}


def test_required_catalog_snapshots(tmp_path):
    args = Namespace(data=graph.DATA, cases=None, runner=None, report=tmp_path / "changed_cases.json", update=False)
    assert graph.run_cases(args) == 0


def test_catalog_maps_every_documented_checkpoint():
    hub = Path(__file__).resolve().parents[4] / "tests/model_hub_tests/gguf"
    if not hub.is_dir():
        pytest.skip("model hub lists are not installed")
    documented = set()
    for name in ("gguf_models_precommit", "gguf_models_nightly", "gguf_mmproj_precommit", "gguf_mmproj_nightly"):
        for line in (hub / name).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                documented.add("/".join(line.split(",")[1:3]))
    documented |= {m["repo_id"] + "/" + m["filename"] for m in graph.load_json(hub / "bonsai_models.json")["models"]}
    mapped = {model for case in graph.load_json(graph.DATA / "catalog.json")["cases"] for model in case["models"]}
    assert sorted(documented - mapped) == [], "Documented checkpoints need a catalog case listing repo/filename"


def test_detects_rewiring_with_unchanged_operation_counts(capsys):
    first = graph.compact_snapshot(snapshot())
    modified = snapshot()
    modified["canonical"]["nodes"][0]["inputs"].reverse()
    second = graph.compact_snapshot(modified)
    before = {"recipe_sha256": "same", "stages": {"deferred_plan": {"models": {"native": first}}}}
    after = {"recipe_sha256": "same", "stages": {"deferred_plan": {"models": {"native": second}}}}
    changes = graph.reference_delta(before, after)
    assert len(changes) == 1 and changes[0]["graph_changed"]
    assert changes[0]["ops"] == []
    graph.print_changes("llama", changes)
    assert "structure/attributes changed; operation counts unchanged" in capsys.readouterr().out
    assert "canonical" not in first


def test_histogram_reports_net_added_and_removed_counts():
    assert graph.histogram_delta([["Add", "opset1", 3], ["MatMul", "opset1", 2]],
                                 [["Add", "opset1", 1], ["Gather", "opset8", 4]]) == [
        {"signature": ["Add", "opset1"], "delta": -2},
        {"signature": ["Gather", "opset8"], "delta": 4},
        {"signature": ["MatMul", "opset1"], "delta": -2},
    ]


@pytest.mark.parametrize("mutate", [
    lambda value: value["tensors"][0].update(shape=[2, 2.5]),
    lambda value: value["tensors"][0].update(storage_shape=[5]),
    lambda value: value["tensors"][0].update(ggml_type=142),
    lambda value: value["metadata"][0].update(type=9, element_type=8),
    lambda value: value["metadata"].append({"name": "epsilon", "type": 6, "value": float("inf")}),
    lambda value: value["metadata"].append(copy.deepcopy(value["metadata"][0])),
    lambda value: value["metadata"].append({"name": "gate", "type": 6, "value": {"nonfinite": "unknown"}}),
])
def test_rejects_invalid_recipe_values_and_layouts(mutate):
    value = recipe()
    mutate(value)
    with pytest.raises(ValueError):
        graph.validate_recipe(value)


def test_typed_metadata_preserves_scalar_and_array_distinctions():
    value = recipe()
    value["metadata"].extend([
        {"name": "scalar", "type": 4, "value": 1},
        {"name": "array", "type": 9, "element_type": 4, "value": [1]},
        {"name": "flag", "type": 7, "value": True},
        {"name": "gate_lower_bound", "type": 6, "value": {"nonfinite": "-inf"}},
    ])
    graph.validate_recipe(value)
    value["metadata"][-2]["value"] = 1
    with pytest.raises(ValueError, match="typed metadata"):
        graph.validate_recipe(value)


def test_import_omits_scores_and_preserves_typed_nonfinite_metadata(tmp_path):
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data
    header = (b"GGUF" + struct.pack("<IQQ", 3, 1, 3) +
              string("general.architecture") + struct.pack("<I", 8) + string("llama") +
              string("tokenizer.ggml.scores") + struct.pack("<IIQf", 9, 6, 1, float("-inf")) +
              string("gate_lower_bound") + struct.pack("<If", 6, float("-inf")) +
              string("weight") + struct.pack("<IQIQ", 1, 1, 0, 0))
    path = tmp_path / "header.gguf"
    path.write_bytes(header + bytes((-len(header)) % 32) + struct.pack("<f", 1))
    imported = graph.import_header(path)
    assert imported["omitted_metadata"] == ["tokenizer.ggml.scores"]
    assert imported["tensors"][0]["storage_shape"] == [1]
    assert len(imported["source_header_sha256"]) == 64
    assert imported["metadata"][-1]["value"] == {"nonfinite": "-inf"}
    path.write_bytes(header)
    assert graph.import_header(path)["tensors"] == imported["tensors"]
    path.write_bytes(path.read_bytes()[:20])
    with pytest.raises(ValueError, match="Truncated"):
        graph.import_header(path)


def test_compact_recipe_roundtrips_layers_and_encoded_arrays(tmp_path):
    value = recipe()
    value["tensors"] = [{"name": name, "ggml_type": kind, "shape": [32, 256],
                         "storage_type": graph.storage(kind, [32, 256])[0],
                         "storage_shape": graph.storage(kind, [32, 256])[1]}
                        for name, kind in [("token_embd.weight", 12), ("blk.0.attn_q.weight", 12),
                                           ("blk.1.attn_q.weight", 14), ("blk.2.attn_q.weight", 12),
                                           ("blk.3.attn_q.weight", 12), ("mm.10.weight", 0)]]
    value["metadata"] += [
        {"name": "eps", "type": 9, "element_type": 6, "value": [0.0] * 70 + [-0.0] * 30},
        {"name": "signs", "type": 9, "element_type": 5, "value": [1, -1, -1] * 25},
        {"name": "gate", "type": 6, "value": {"nonfinite": "-inf"}},
    ]
    path = tmp_path / "recipe.json"
    path.write_text(graph.recipe_text(value))
    document = json.loads(path.read_text())
    assert ["blk.{}.attn_q.weight", "Q4_K", [32, 256], "0,2-3"] in document["tensors"]
    assert ["mm.10.weight", "F32", [32, 256]] in document["tensors"]
    assert document["metadata"]["eps"][1] == {"rle": [[0.0, 70], [-0.0, 30]]}
    assert set(document["metadata"]["signs"][1]) == {"signs", "count"}
    decoded = graph.load_recipe(path)
    assert graph.canonical_bytes(decoded["metadata"]) == graph.canonical_bytes(value["metadata"])
    assert sorted(decoded["tensors"], key=lambda t: t["name"]) == sorted(value["tensors"], key=lambda t: t["name"])


@pytest.mark.parametrize("tensor", [
    ["blk.{}.w", "F32", [2], "3-1"],
    ["blk.{}.w", "F32", [2], "0-2,2"],
    ["blk.{}.w", "F32", [2]],
    ["blk.0.w", "Q9_K", [2]],
])
def test_compact_recipe_rejects_invalid_tensor_groups(tensor):
    document = {"version": 2, "metadata": {"general.architecture": ["str", "llama"]}, "tensors": [tensor]}
    with pytest.raises(ValueError):
        graph.decode_recipe(document)


def test_reference_size_ceiling_is_bytes_not_node_count():
    graph.reference_bytes({"diagnostic": "x" * (graph.MAX_REFERENCE_BYTES - 32)})
    with pytest.raises(ValueError, match="32 KiB"):
        graph.reference_bytes({"diagnostic": "x" * graph.MAX_REFERENCE_BYTES})


def test_reference_format_keeps_histogram_rows_diffable():
    value = {"version": 1, "recipe_sha256": "0" * 64,
             "stages": {"deferred_plan": graph.model_stage({"native": graph.compact_snapshot(snapshot())})}}
    data = graph.reference_bytes(value).decode()
    assert '["MatMul","opset1",1]' in data
    assert json.loads(data) == value


def test_update_requires_selection_and_finishes_capture_before_writing(tmp_path, monkeypatch):
    value = recipe()
    original = json.dumps({"version": 1, "recipe_sha256": "0" * 64,
                           "stages": {"deferred_plan": graph.model_stage(
                               {"native": graph.compact_snapshot(snapshot())})}})
    cases = []
    for name in ("one", "two"):
        (tmp_path / f"{name}.recipe.json").write_text(graph.recipe_text(value))
        (tmp_path / f"{name}.reference.json").write_text(original)
        cases.append({"id": name, "recipe": f"{name}.recipe.json", "reference": f"{name}.reference.json",
                      "models": [name], "architecture": "llama",
                      "source": {"filename": "test.gguf", "header_sha256": "0" * 64}})
    (tmp_path / "catalog.json").write_text(json.dumps({"version": 1, "cases": cases}))
    args = Namespace(data=tmp_path, cases=None, runner="runner", report=tmp_path / "changed_cases.json", update=True)
    with pytest.raises(ValueError, match="explicit --cases"):
        graph.run_cases(args)
    args.cases = "missing"
    with pytest.raises(ValueError, match="Unknown/empty"):
        graph.run_cases(args)
    def capture(path, runner):
        if Path(path).name.startswith("two"):
            raise ValueError("intentional second-case failure")
        return graph.compact_snapshot(snapshot())
    monkeypatch.setattr(graph, "capture", capture)
    args.cases = "one,two"
    with pytest.raises(ValueError, match="second-case failure"):
        graph.run_cases(args)
    assert all((tmp_path / case["reference"]).read_text() == original for case in cases)
    report = json.loads(args.report.read_text())
    assert report["selected"] == 2 and report["completed"] == 1
    assert "second-case failure" in report["errors"][0]
    monkeypatch.setattr(graph, "capture", lambda path, runner: graph.compact_snapshot(snapshot()))
    replace = graph.os.replace

    def fail_second_publish(source, target):
        if Path(target).name == "two.reference.json":
            raise OSError("intentional publication failure")
        replace(source, target)

    monkeypatch.setattr(graph.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="publication failure"):
        graph.run_cases(args)
    assert all((tmp_path / case["reference"]).read_text() == original for case in cases)
    calls = 0

    def fail_publish_and_rollback(source, target):
        nonlocal calls
        calls += 1
        if calls in (2, 3):
            raise OSError("intentional rollback failure")
        replace(source, target)

    monkeypatch.setattr(graph.os, "replace", fail_publish_and_rollback)
    with pytest.raises(OSError, match="retained backups"):
        graph.run_cases(args)
    backups = [path for path in tmp_path.iterdir() if path.name.startswith("tmp")]
    assert len(backups) == 1 and backups[0].read_text() == original
    backups[0].replace(tmp_path / "one.reference.json")
    cases[1]["reference"] = "./one.reference.json"
    (tmp_path / "catalog.json").write_text(json.dumps({"version": 1, "cases": cases}))
    args.cases = "one"
    with pytest.raises(ValueError, match="Duplicate reference target"):
        graph.run_cases(args)


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(version=True),
    lambda value: value["stages"]["deferred_plan"].pop("map_sha256"),
    lambda value: value["stages"]["deferred_plan"]["models"]["native"].update(graph_sha256="short"),
    lambda value: value["stages"].update(unknown={}),
])
def test_reference_schema_rejects_malformed_stages(mutate):
    value = {"version": 1, "recipe_sha256": "0" * 64,
             "stages": {"deferred_plan": graph.model_stage({"native": graph.compact_snapshot(snapshot())})}}
    mutate(value)
    with pytest.raises(ValueError):
        graph.validate_document(value, "reference")


def test_materialized_recipe_roundtrip_and_budget(tmp_path):
    value = recipe()
    value["tensors"][0].update(shape=[2, 3], storage_shape=[2, 3])
    value["payload"] = {"mode": "pattern", "max_bytes": 32}
    path = tmp_path / "synthetic.gguf"
    graph.materialize_recipe(value, path)
    assert graph.import_header(path)["tensors"] == value["tensors"]
    value["payload"]["max_bytes"] = 4
    with pytest.raises(ValueError, match="budget"):
        graph.materialize_recipe(value, path)


@pytest.mark.parametrize("kind", [3, 12, 39, 42, 142, 143])
def test_packed_zero_payloads_preserve_block_layout(tmp_path, kind):
    value = recipe()
    block, size = graph.BLOCKS[kind]
    value["tensors"][0].update(shape=[2, 256], storage_type="u8", storage_shape=[512 // block * size],
                              ggml_type=kind)
    value["payload"] = {"mode": "zeros", "max_bytes": 4096}
    path = tmp_path / "synthetic.gguf"
    graph.materialize_recipe(value, path)
    assert graph.import_header(path)["tensors"] == value["tensors"]
    if kind == 42:
        prefix = tmp_path / "prefix.gguf"
        prefix.write_bytes(path.read_bytes()[:-value["tensors"][0]["storage_shape"][0] // 2])
        with pytest.raises(ValueError):
            graph.import_header(prefix)
        assert graph.import_header(prefix, path.stat().st_size)["tensors"] == value["tensors"]
    value["payload"]["mode"] = "pattern"
    with pytest.raises(ValueError, match="explicit zeros"):
        graph.materialize_recipe(value, path)
