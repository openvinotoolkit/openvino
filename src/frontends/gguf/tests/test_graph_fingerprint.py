# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Required recipe gate and focused standard-library tooling contracts."""

from argparse import Namespace
import copy
import json
from pathlib import Path
import struct
from subprocess import CompletedProcess

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


def captured_models():
    return {"models": {"ids/language": graph.compact_snapshot(snapshot())}, "projectors": []}


def reference():
    return {"version": graph.REFERENCE_VERSION, **captured_models()}


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


@pytest.mark.parametrize(("case_id", "has_embeddings"), [
    ("representative--llama-ggml-2", False),
    ("llama--llava-v1.6-vicuna-7b.q4_k_m--c28f6ad7", True),
    ("llama--ggml-model-q4_k--2ccee2f3", True),
    ("qwen2--ggml-model-q4_0--208e9dab", True),
    ("gemma3--gemma-3-1b-it-q4_0--9af9428e", False),
    ("gemma3--gemma-3-4b-it-q4_k_m--b6f02999", True),
    ("gemma4--nemotron-3.5-bolt-e4b-q4_k_m--b24b6ba5", False),
    ("gemma4--gemma-4-e4b-it-q4_0--e7c11bc9", True),
    ("lfm3--lfm3-1.7b-qad-q4_0--d897a7f6", False),
    ("lfm3--lfm3-audio-2b-q4_0--07f0bced", True),
    ("deepseek2-ocr--deepseek-ocr-2-q4_k_m--e6ed4c0d", True),
    ("llama--devstral-small-2507-q4_k_m--0bc56937", False),
    ("mistral3--devstral-small-2-24b-instruct-2512-q4_k_m--d09200fa", True),
    ("muse-glimmer--muse-glimmer-30b-q4_0--5bbf61da", True),
    ("qwen35--ternary-bonsai-2-2b-pq2_0--fd8dc1f3", True),
    ("qwen35--ornith-1.5-9b-q4_k_m--39b1a33b", True),
])
def test_catalog_distinguishes_text_and_multimodal_cases(case_id, has_embeddings):
    case = next(case for case in graph.load_json(graph.DATA / "catalog.json")["cases"] if case["id"] == case_id)
    modes = ["ids", "embeds"] if has_embeddings else ["ids"]
    assert case["input_modes"] == modes
    models = graph.load_json(graph.DATA / case["reference"])["models"]
    assert ("embeds/language" in models) == has_embeddings
    assert "ids/language" in models


@pytest.mark.parametrize("changes", [
    {},
    {"input_modes": ["embeds"]},
    {"input_modes": ["ids", "ids"]},
    {"architecture": "clip", "input_modes": ["ids"]},
    {"native_only": "unsupported adapter", "input_modes": ["ids"]},
])
def test_catalog_rejects_missing_or_inapplicable_input_modes(tmp_path, changes):
    case = {"id": "test", "architecture": "llama", "models": ["test"], "recipe": "recipe.json",
            "reference": "reference.json", "source": {"filename": "test.gguf", "header_sha256": "0" * 64}}
    case.update(changes)
    (tmp_path / "catalog.json").write_text(json.dumps({"version": 1, "cases": [case]}))
    args = Namespace(data=tmp_path, cases=None, runner="unused", report=tmp_path / "report.json", update=False)
    with pytest.raises(ValueError, match="input_modes"):
        graph.run_cases(args)


def test_catalog_rejects_unlisted_equivalent_source_models(tmp_path):
    source = {"filename": "test.gguf", "header_sha256": "0" * 64}
    case = {"id": "test", "architecture": "llama", "input_modes": ["ids"], "models": ["test"],
            "recipe": "recipe.json", "reference": "reference.json", "source": source,
            "equivalent_sources": [{"models": ["unlisted"], "source": source}]}
    (tmp_path / "catalog.json").write_text(json.dumps({"version": 1, "cases": [case]}))
    args = Namespace(data=tmp_path, cases=None, runner="unused", report=tmp_path / "report.json", update=False)
    with pytest.raises(ValueError, match="equivalent source models"):
        graph.run_cases(args)


def test_decoder_capture_uses_selected_input_modes():
    cases = graph.load_json(graph.DATA / "catalog.json")["cases"]
    case = next(case for case in cases if case["id"] == "fixture--llama-dense")
    value = graph.load_recipe(graph.DATA / case["recipe"])
    runner = graph.runner_path(None)

    value["input_modes"] = ["ids"]
    ids = graph.capture(value, runner)
    assert set(ids["models"]) == {"ids/language"}

    value["input_modes"] = ["ids", "embeds"]
    both = graph.capture(value, runner)
    assert set(both["models"]) == {"ids/language", "embeds/language", "embeds/text_embeddings"}
    assert ids["models"]["ids/language"] == both["models"]["ids/language"]

    value.pop("input_modes")
    assert graph.capture(value, runner) == both

    value["input_modes"] = ["embeds"]
    with pytest.raises(ValueError, match="Invalid decoder input modes"):
        graph.capture(value, runner)


def test_draft_vocabulary_map_requires_i64_storage():
    cases = graph.load_json(graph.DATA / "catalog.json")["cases"]
    case = next(case for case in cases if case["architecture"] == "eagle3")
    value = graph.load_recipe(graph.DATA / case["recipe"])
    value["input_modes"] = case["input_modes"]
    mapping = next(tensor for tensor in value["tensors"] if tensor["name"] == "d2t")
    mapping["storage_type"] = "f32"
    with pytest.raises(ValueError, match="Invalid synthetic draft vocabulary map descriptor"):
        graph.capture(value, graph.runner_path(None))


def test_detects_rewiring_with_unchanged_operation_counts(capsys):
    first = graph.compact_snapshot(snapshot())
    modified = snapshot()
    modified["canonical"]["nodes"][0]["inputs"].reverse()
    second = graph.compact_snapshot(modified)
    expected, actual = reference(), reference()
    expected["models"]["ids/language"] = first
    actual["models"]["ids/language"] = second
    change = graph.reference_delta(expected, actual)
    assert change["models"]["ids/language"]["graph_changed"] and change["models"]["ids/language"]["ops"] == []
    assert graph.reference_delta(expected, expected) is None
    graph.print_changes("llama", change)
    assert "structure/attributes changed; operation counts unchanged" in capsys.readouterr().out
    assert "canonical" not in first


def test_groups_each_named_graph_by_changed_operation_types_regardless_of_depth(capsys):
    def delta(ops):
        return {"status": "changed", "before_nodes": 10, "after_nodes": 10 + sum(ops.values()),
                "graph_changed": True, "projections": [],
                "ops": [{"signature": [op, "opset1"], "delta": count} for op, count in ops.items()]}

    def change(name, graphs, projectors_changed=False):
        return {"case": name, "architecture": name.split("--")[0], "models": [name],
                "change": {"projectors_changed": projectors_changed,
                           "models": {graph_name: delta(ops) for graph_name, ops in graphs.items()}}}
    changes = [change("representative--llama-ggml-2", {"ids/language": {"Multiply": 4}}),
               change("llama--small", {"ids/language": {"Multiply": 16}}),
               change("qwen3--deep", {"ids/language": {"Multiply": 36}}),
               change("qwen3--removed", {"ids/language": {"Multiply": -36}}),
               change("gemma3--multimodal", {"ids/language": {"Multiply": 8}, "embeds/language": {"Add": 8},
                                             "embeds/text_embeddings": {}, "vision": {}}, True)]
    groups = graph.group_changes(changes)
    assert [[graph.member_label(member) for member in group["members"]] for group in groups] == [
        ["representative--llama-ggml-2 [ids/language]", "llama--small [ids/language]",
         "qwen3--deep [ids/language]", "gemma3--multimodal [ids/language]"],
        ["qwen3--removed [ids/language]"], ["gemma3--multimodal [embeds/language]"],
        ["gemma3--multimodal [embeds/text_embeddings]"], ["gemma3--multimodal [vision]"],
        ["gemma3--multimodal [projectors]"]]
    assert groups[0]["representative"] == {"case": "llama--small", "graph": "ids/language"}
    assert groups[0]["architectures"] == ["gemma3", "llama", "qwen3", "representative"]
    graph.print_groups(groups, limit=2)
    output = capsys.readouterr().out
    assert "Change group 1: 4 graph(s) in 4 case(s)" in output and "... and 2 more" in output
    assert "Representative (deltas shown for this graph): llama--small [ids/language]" in output
    assert "gemma3--multimodal: projector identities changed" in output


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
    header = (b"GGUF" + struct.pack("<IQQ", 3, 1, 4) +
              string("general.architecture") + struct.pack("<I", 8) + string("llama") +
              string("tokenizer.ggml.scores") + struct.pack("<IIQf", 9, 6, 1, float("-inf")) +
              string("gate_lower_bound") + struct.pack("<If", 6, float("-inf")) +
              string("general.tags") + struct.pack("<IIQ", 9, 8, 1) + string("automatic-speech-recognition") +
              string("weight") + struct.pack("<IQIQ", 1, 1, 0, 0))
    path = tmp_path / "header.gguf"
    path.write_bytes(header + bytes((-len(header)) % 32) + struct.pack("<f", 1))
    imported = graph.import_header(path)
    assert imported["omitted_metadata"] == ["tokenizer.ggml.scores"]
    assert imported["tensors"][0]["storage_shape"] == [1]
    assert len(graph.import_header_with_digest(path)[1]) == 64
    assert imported["metadata"][-2]["value"] == {"nonfinite": "-inf"}
    assert imported["metadata"][-1]["value"] == ["automatic-speech-recognition"]
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
    value = reference()
    data = graph.reference_bytes(value).decode()
    assert '["MatMul","opset1",1]' in data
    assert json.loads(data) == value


def test_update_requires_selection_and_finishes_capture_before_writing(tmp_path, monkeypatch, capsys):
    value = recipe()
    original = json.dumps(reference())
    cases = []
    for name in ("one", "two"):
        value["tensors"][0]["name"] = name
        (tmp_path / f"{name}.recipe.json").write_text(graph.recipe_text(value))
        (tmp_path / f"{name}.reference.json").write_text(original)
        cases.append({"id": name, "recipe": f"{name}.recipe.json", "reference": f"{name}.reference.json",
                      "models": [name], "architecture": "llama", "input_modes": ["ids"],
                      "source": {"filename": "test.gguf", "header_sha256": "0" * 64}})
    (tmp_path / "catalog.json").write_text(json.dumps({"version": 1, "cases": cases}))
    args = Namespace(data=tmp_path, cases=None, runner="runner", report=tmp_path / "changed_cases.json", update=True)
    with pytest.raises(ValueError, match="explicit --cases"):
        graph.run_cases(args)
    args.cases = "missing"
    with pytest.raises(ValueError, match="Unknown/empty"):
        graph.run_cases(args)
    def capture(value, runner):
        if value["tensors"][0]["name"] == "two":
            raise ValueError("intentional second-case failure")
        result = captured_models()
        result["models"]["ids/language"]["graph_sha256"] = "f" * 64
        return result
    monkeypatch.setattr(graph, "capture", capture)
    args.cases = "one,two"
    with pytest.raises(ValueError, match="second-case failure"):
        graph.run_cases(args)
    assert all((tmp_path / case["reference"]).read_text() == original for case in cases)
    report = json.loads(args.report.read_text())
    assert report["selected"] == 2 and report["completed"] == 1
    assert "second-case failure" in report["errors"][0]
    assert report["change_groups"][0]["members"] == [{"case": "one", "graph": "ids/language"}]
    assert "one [ids/language]: changed" in capsys.readouterr().out
    monkeypatch.setattr(graph, "capture", lambda value, runner: captured_models())
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
    lambda value: value.update(version=1),
    lambda value: value.update(recipe_sha256="0" * 64),
    lambda value: value["models"]["ids/language"].update(graph_sha256="short"),
    lambda value: value.update(stages={}),
    lambda value: value["models"]["ids/language"].pop("ops"),
    lambda value: value.update(models={}),
    lambda value: value["models"].update({"": graph.compact_snapshot(snapshot())}),
])
def test_reference_schema_rejects_malformed_snapshots(mutate):
    value = reference()
    graph.validate_document(value, "reference")
    mutate(value)
    with pytest.raises(ValueError):
        graph.validate_document(value, "reference")


def test_import_accepts_packed_header_prefix_with_source_size(tmp_path):
    block, size = graph.BLOCKS[42]
    payload = 512 // block * size
    name = b"weight"
    key, arch = b"general.architecture", b"llama"
    header = (b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + struct.pack("<Q", len(key)) + key +
              struct.pack("<IQ", 8, len(arch)) + arch + struct.pack("<Q", len(name)) + name +
              struct.pack("<IQQIQ", 2, 256, 2, 42, 0))
    header += bytes((-len(header)) % 32)
    prefix = tmp_path / "prefix.gguf"
    prefix.write_bytes(header + bytes(payload // 2))
    with pytest.raises(ValueError):
        graph.import_header(prefix)
    tensors = graph.import_header(prefix, len(header) + payload)["tensors"]
    assert tensors == [{"name": "weight", "shape": [2, 256], "ggml_type": 42,
                        "storage_type": "u8", "storage_shape": [payload]}]


def test_check_failure_explains_update_and_evidence_in_ci(tmp_path, monkeypatch, capsys):
    reference_text = json.dumps(reference())
    cases = []
    for name in ("same", "changed"):
        value = recipe()
        value["tensors"][0]["name"] = name
        (tmp_path / f"{name}.recipe.json").write_text(graph.recipe_text(value))
        (tmp_path / f"{name}.reference.json").write_text(reference_text)
        cases.append({"id": name, "recipe": f"{name}.recipe.json", "reference": f"{name}.reference.json",
                      "models": [name], "architecture": "llama", "input_modes": ["ids"],
                      "source": {"filename": "test.gguf", "header_sha256": "0" * 64}})
    (tmp_path / "catalog.json").write_text(json.dumps({"version": 1, "cases": cases}))

    def capture(value, runner):
        result = captured_models()
        if value["tensors"][0]["name"] == "changed":
            result["models"]["ids/language"]["graph_sha256"] = "f" * 64
        return result
    monkeypatch.setattr(graph, "capture", capture)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = Namespace(data=tmp_path, cases=None, runner="runner", report=tmp_path / "changed_cases.json", update=False)

    assert graph.run_cases(args) == 1
    out = capsys.readouterr().out
    command = f"python3 {graph.SCRIPT_PATH} --cases changed --update"
    policy = graph.README_PATH + graph.POLICY_ANCHOR
    assert command in out and policy in out and graph.RUNNER_EXPORT in out
    assert "::error title=GGUF structural references changed::" in out
    text = summary.read_text()
    assert command in text and policy in text and graph.RUNNER_EXPORT in text
    assert "| 1 | 1 | 1 | `changed` | `ids/language` | changed |" in text
    assert "structure/attributes only" in text and "`same`" not in text
    assert "- `changed` `[ids/language]`" in text
    report = json.loads(args.report.read_text())
    assert report["change_groups"][0]["members"] == [{"case": "changed", "graph": "ids/language"}]
    assert "does not prove equivalent" in out and "does not prove equivalent" in text


def test_compares_every_named_model_and_detects_added_removed_and_renamed_components(capsys):
    expected = reference()
    expected["models"].update({name: graph.compact_snapshot(snapshot()) for name in
                              ("embeds/language", "embeds/text_embeddings", "embeds/text_embeddings_per_layer")})
    actual = copy.deepcopy(expected)
    actual["models"]["embeds/text_embeddings_per_layer"]["graph_sha256"] = "a" * 64
    actual["models"]["embeds/renamed"] = actual["models"].pop("embeds/text_embeddings")
    delta = graph.reference_delta(expected, actual)
    assert set(delta["models"]) == {"embeds/text_embeddings_per_layer", "embeds/text_embeddings", "embeds/renamed"}
    assert delta["models"]["embeds/text_embeddings"]["status"] == "removed"
    assert delta["models"]["embeds/renamed"]["status"] == "added"
    graph.print_changes("gemma4", delta)
    output = capsys.readouterr().out
    assert "[embeds/text_embeddings_per_layer]: changed" in output
    assert "[embeds/text_embeddings]: removed" in output
    assert "[embeds/renamed]: added" in output


def test_capture_compacts_all_returned_models_and_preserves_projector_identity(monkeypatch):
    models = {"vision_embeddings": snapshot(), "vision_embeddings_pos": snapshot(),
              "vision_embeddings_merger": snapshot(), "audio_generation_codec": snapshot()}
    models["vision_embeddings_merger"]["canonical"]["nodes"][0]["inputs"].reverse()

    def run(command, **kwargs):
        Path(command[4]).write_text(json.dumps({"models": models, "projectors": ["vision.qwen3vl_merger"]}))
        return CompletedProcess(command, 0)

    monkeypatch.setattr(graph.subprocess, "run", run)
    actual = graph.capture(recipe(), "runner")
    assert set(actual["models"]) == set(models)
    assert actual["projectors"] == ["vision.qwen3vl_merger"]
    assert actual["models"]["vision_embeddings"]["graph_sha256"] != \
        actual["models"]["vision_embeddings_merger"]["graph_sha256"]
    assert all("canonical" not in model for model in actual["models"].values())


def test_capture_rejects_empty_model_map(monkeypatch):
    def run(command, **kwargs):
        Path(command[4]).write_text('{"models":{},"projectors":[]}')
        return CompletedProcess(command, 0)

    monkeypatch.setattr(graph.subprocess, "run", run)
    with pytest.raises(ValueError, match="nonempty named model map"):
        graph.capture(recipe(), "runner")
