# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# What: unit tests for run_multi_env.py's pure logic: manifest scope/run filtering, pytest command
#       construction from a [[runs]] entry, JUnit-XML merging, and exit-code aggregation.
# How: uses small synthetic manifest dicts / XML fixtures instead of the real pytorch_models.toml or
#      real pytest runs, so these run fast and don't need any venv or network access.
# Why: run_multi_env.py is the single entry point CI calls for every model_scope; a bug in run
#      selection, command construction, or exit-code aggregation would silently skip or misreport
#      whole test suites without any test ever actually failing.
#
# Run in precommit by pytorch_models.toml's `multi_env_unit` run; or directly, e.g.:
#   PYTHONPATH=tests/model_hub_tests <venv>/bin/python -m pytest
#       tests/model_hub_tests/multi_env/tests/test_run_multi_env.py -q
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_multi_env as rme  # noqa: E402

pytestmark = pytest.mark.precommit


def _sample_manifest():
    return {
        "envs": {
            "envA": {"steps": [["reqs/a.txt"]]},
            "envB": {"steps": [["reqs/b.txt"]]},
        },
        "runs": [
            {
                "name": "run_precommit_only",
                "scopes": ["precommit"],
                "paths": ["../pytorch"],
                "default_env": "envA",
                "envs": ["envA"],
            },
            {
                "name": "run_both_nightly",
                "scopes": ["precommit", "nightly_scope2"],
                "paths": ["../pytorch/test_llm.py"],
                "default_env": "envA",
                "envs": ["envA", "envB"],
                "xdist_n": 2,
                "html_name": "TEST-{scope}.html",
                "extra_env": {"OP_REPORT_FILE": "unsupported_ops.log"},
            },
            {
                "name": "run_nightly1_only",
                "scopes": ["nightly_scope1"],
                "paths": ["../pytorch"],
                "default_env": "envA",
                "envs": ["envA"],
                "pytest_args": ["-m", "nightly"],
            },
        ],
    }


# --- select_runs / build_pairs ---

def test_select_runs_filters_by_scope():
    manifest = _sample_manifest()
    runs = rme.select_runs(manifest, "precommit")
    assert [r["name"] for r in runs] == ["run_precommit_only", "run_both_nightly"]


def test_select_runs_disjoint_nightly_scopes():
    manifest = _sample_manifest()
    assert [r["name"] for r in rme.select_runs(manifest, "nightly_scope1")] == ["run_nightly1_only"]
    assert [r["name"] for r in rme.select_runs(manifest, "nightly_scope2")] == ["run_both_nightly"]


def test_select_runs_filters_by_name():
    manifest = _sample_manifest()
    runs = rme.select_runs(manifest, "precommit", only_names=["run_both_nightly"])
    assert [r["name"] for r in runs] == ["run_both_nightly"]


def test_select_runs_no_match_returns_empty():
    manifest = _sample_manifest()
    assert rme.select_runs(manifest, "precommit", only_names=["does_not_exist"]) == []


def test_build_pairs_expands_one_pair_per_env():
    manifest = _sample_manifest()
    runs = rme.select_runs(manifest, "precommit")
    pairs = rme.build_pairs(runs)
    assert pairs == [
        (runs[0], "envA"),
        (runs[1], "envA"),
        (runs[1], "envB"),
    ]


# --- build_pytest_invocation ---

def test_invocation_basic_marker_and_paths(tmp_path):
    manifest = _sample_manifest()
    run = manifest["runs"][0]
    manifest_dir = tmp_path / "multi_env"
    manifest_dir.mkdir()
    results_dir = tmp_path / "results"

    inv = rme.build_pytest_invocation("/venv/bin/python", run, "envA", "precommit",
                                       manifest_dir, results_dir)

    assert inv.cmd[0] == "/venv/bin/python"
    assert inv.cmd[1:3] == ["-m", "pytest"]  # `python -m pytest`
    assert str((manifest_dir / "../pytorch").resolve()) in inv.cmd
    marker_flag_index = inv.cmd.index("-m", 2)  # the pytest marker flag, not `python -m`
    assert inv.cmd[marker_flag_index + 1] == "precommit"
    assert inv.env_vars["OV_TEST_ENV"] == "envA"
    assert inv.env_vars["OV_TEST_DEFAULT_ENV"] == "envA"
    assert inv.env_vars["TEST_DEVICE"] == "CPU"
    assert inv.env_vars["PYTHONPATH"].startswith(str(manifest_dir.parent))
    # single-env run: no "_<env>" suffix, this file IS the combined result (see main()'s merge skip)
    assert inv.junit_path == results_dir / "TEST-torch_run_precommit_only.junit.xml"


def test_invocation_respects_existing_markexpr_in_pytest_args(tmp_path):
    manifest = _sample_manifest()
    run = manifest["runs"][2]  # run_nightly1_only, pytest_args already has -m nightly
    inv = rme.build_pytest_invocation("/venv/bin/python", run, "envA", "nightly_scope1",
                                       tmp_path / "multi_env", tmp_path / "results")
    # only ONE `-m` after `python -m pytest`'s own -- i.e. we did not append a second, redundant
    # marker flag on top of the run's own pytest_args.
    assert inv.cmd.count("-m") == 2
    marker_flag_index = inv.cmd.index("-m", 2)
    assert inv.cmd[marker_flag_index + 1] == "nightly"


def test_invocation_xdist_n_added(tmp_path):
    manifest = _sample_manifest()
    run = manifest["runs"][1]
    inv = rme.build_pytest_invocation("/venv/bin/python", run, "envA", "precommit",
                                       tmp_path / "multi_env", tmp_path / "results")
    assert "-n" in inv.cmd and inv.cmd[inv.cmd.index("-n") + 1] == "2"


def test_invocation_html_disambiguated_per_env_when_multiple_envs(tmp_path):
    manifest = _sample_manifest()
    run = manifest["runs"][1]  # envs = [envA, envB]
    inv_a = rme.build_pytest_invocation("/venv/bin/python", run, "envA", "precommit",
                                         tmp_path / "multi_env", tmp_path / "results")
    inv_b = rme.build_pytest_invocation("/venv/bin/python", run, "envB", "precommit",
                                         tmp_path / "multi_env", tmp_path / "results")
    html_a = next(a for a in inv_a.cmd if a.startswith("--html="))
    html_b = next(a for a in inv_b.cmd if a.startswith("--html="))
    assert html_a != html_b
    assert "envA" in html_a and "envB" in html_b


def test_invocation_junit_filename_disambiguated_per_env_when_multiple_envs(tmp_path):
    manifest = _sample_manifest()
    run = manifest["runs"][1]  # envs = [envA, envB]
    inv_a = rme.build_pytest_invocation("/venv/bin/python", run, "envA", "precommit",
                                         tmp_path / "multi_env", tmp_path / "results")
    inv_b = rme.build_pytest_invocation("/venv/bin/python", run, "envB", "precommit",
                                         tmp_path / "multi_env", tmp_path / "results")
    assert inv_a.junit_path != inv_b.junit_path
    assert inv_a.junit_path.name == "TEST-torch_run_both_nightly_envA.junit.xml"


def test_invocation_html_not_disambiguated_for_single_env_run(tmp_path):
    manifest = _sample_manifest()
    run = manifest["runs"][0]  # no html_name at all -> no --html flag
    inv = rme.build_pytest_invocation("/venv/bin/python", run, "envA", "precommit",
                                       tmp_path / "multi_env", tmp_path / "results")
    assert not any(a.startswith("--html=") for a in inv.cmd)


def test_invocation_extra_env_relative_filename_joined_with_results_dir(tmp_path):
    manifest = _sample_manifest()
    run = manifest["runs"][1]
    results_dir = tmp_path / "results"
    inv = rme.build_pytest_invocation("/venv/bin/python", run, "envA", "precommit",
                                       tmp_path / "multi_env", results_dir)
    assert inv.env_vars["OP_REPORT_FILE"] == str(results_dir / "unsupported_ops.log")


def test_invocation_activates_venv_path_and_virtual_env(tmp_path):
    manifest = _sample_manifest()
    run = manifest["runs"][0]
    inv = rme.build_pytest_invocation("/some/venv/bin/python", run, "envA", "precommit",
                                       tmp_path / "multi_env", tmp_path / "results")
    assert inv.env_vars["VIRTUAL_ENV"] == "/some/venv"
    assert inv.env_vars["PATH"].startswith("/some/venv/bin" + os.pathsep)


def test_invocation_activates_symlinked_venv_bin_not_its_resolved_target(tmp_path):
    # A real venv's bin/python is a symlink (bin/python -> python3.X -> /usr/bin/python3.X). resolve()
    # follows the whole chain, landing on the system interpreter's own directory instead of the venv's
    # bin/ -- exactly the bug that broke test_aliked.py (a bare `python` on PATH resolved outside the
    # venv, so its custom-op build script ran against the wrong interpreter and silently failed without
    # torch). absolute() must be used instead, since it normalizes without dereferencing.
    real_target_dir = tmp_path / "usr_bin"
    real_target_dir.mkdir()
    (real_target_dir / "python3.11").touch()
    venv_bin = tmp_path / "myvenv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(real_target_dir / "python3.11")

    manifest = _sample_manifest()
    run = manifest["runs"][0]
    inv = rme.build_pytest_invocation(str(venv_bin / "python"), run, "envA", "precommit",
                                       tmp_path / "multi_env", tmp_path / "results")
    assert inv.env_vars["VIRTUAL_ENV"] == str(tmp_path / "myvenv")
    assert inv.env_vars["PATH"].startswith(str(venv_bin) + os.pathsep)
    assert not inv.env_vars["PATH"].startswith(str(real_target_dir) + os.pathsep)


def test_invocation_extra_env_absolute_passthrough(tmp_path):
    manifest = _sample_manifest()
    run = dict(manifest["runs"][1])
    run["extra_env"] = {"OP_REPORT_FILE": "/abs/path/report.log"}
    inv = rme.build_pytest_invocation("/venv/bin/python", run, "envA", "precommit",
                                       tmp_path / "multi_env", tmp_path / "results")
    assert inv.env_vars["OP_REPORT_FILE"] == "/abs/path/report.log"


# --- exit code aggregation ---

def test_classify_exit_code_ok_values():
    assert rme.classify_exit_code(0)


def test_classify_exit_code_failure_values():
    # 5 ("no tests collected") is deliberately NOT treated as success: an env/run pair collecting
    # zero tests means the env: tags or the manifest drifted apart, which should fail loudly rather
    # than silently pass.
    assert not rme.classify_exit_code(5)
    assert not rme.classify_exit_code(1)
    assert not rme.classify_exit_code(2)
    assert not rme.classify_exit_code(-9)


def test_aggregate_exit_codes_all_ok():
    assert rme.aggregate_exit_codes([0, 0, 0]) == 0


def test_aggregate_exit_codes_one_failure_propagates():
    assert rme.aggregate_exit_codes([0, 5, 1]) == 1


def test_aggregate_exit_codes_empty_is_ok():
    assert rme.aggregate_exit_codes([]) == 0


# --- merge_junit_files ---

_SUITE_1 = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="2" failures="0" errors="0" skipped="0" time="1.5">
  <testcase classname="a" name="t1" time="0.5"/>
  <testcase classname="a" name="t2" time="1.0"/>
</testsuite></testsuites>
"""

# Older pytest versions emit a bare <testsuite> root (no <testsuites> wrapper) -- merge must accept both.
_SUITE_2 = """<testsuite name="pytest" tests="1" failures="1" errors="0" skipped="0" time="0.25">
  <testcase classname="b" name="t3" time="0.25"><failure message="boom">boom</failure></testcase>
</testsuite>
"""


def test_merge_junit_files_sums_counters_and_concatenates_testcases(tmp_path):
    f1 = tmp_path / "envA.junit.xml"
    f2 = tmp_path / "envB.junit.xml"
    f1.write_text(_SUITE_1)
    f2.write_text(_SUITE_2)

    tree, missing = rme.merge_junit_files([f1, f2], "myrun")

    assert missing == []
    testsuite = tree.getroot().find("testsuite")
    assert testsuite.get("name") == "myrun"
    assert testsuite.get("tests") == "3"
    assert testsuite.get("failures") == "1"
    assert testsuite.get("errors") == "0"
    assert testsuite.get("skipped") == "0"
    assert float(testsuite.get("time")) == 1.75
    names = [tc.get("name") for tc in testsuite.findall("testcase")]
    assert names == ["t1", "t2", "t3"]


def test_merge_junit_files_reports_missing_without_failing(tmp_path):
    f1 = tmp_path / "envA.junit.xml"
    f1.write_text(_SUITE_1)
    missing_path = tmp_path / "does_not_exist.xml"

    tree, missing = rme.merge_junit_files([f1, missing_path], "myrun")

    assert missing == [str(missing_path)]
    testsuite = tree.getroot().find("testsuite")
    assert testsuite.get("tests") == "2"


def test_merge_junit_files_output_is_well_formed_xml(tmp_path):
    f1 = tmp_path / "envA.junit.xml"
    f1.write_text(_SUITE_1)
    out = tmp_path / "combined.xml"

    tree, _ = rme.merge_junit_files([f1], "myrun")
    tree.write(str(out), encoding="UTF-8", xml_declaration=True)

    # round-trips through a fresh parse without error
    reparsed = ET.parse(out)
    assert reparsed.getroot().tag == "testsuites"


# --- main(): one env's build failure must not take down every other (run, env) pair ---

_MANIFEST_TOML = """
[envs.good_env]
steps = [["reqs.txt"]]

[envs.bad_env]
steps = [["reqs.txt"]]

[[runs]]
name = "good_run"
scopes = ["precommit"]
paths = ["dummy_test.py"]
default_env = "good_env"
envs = ["good_env"]

[[runs]]
name = "bad_run"
scopes = ["precommit"]
paths = ["dummy_test.py"]
default_env = "bad_env"
envs = ["bad_env"]
"""


def test_main_isolates_one_envs_build_failure_from_the_rest(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "multi_env"
    manifest_dir.mkdir()
    (manifest_dir / "reqs.txt").write_text("")
    manifest_path = manifest_dir / "pytorch_models.toml"
    manifest_path.write_text(_MANIFEST_TOML)

    def fake_build_env(name, manifest_dir, steps_raw, venvs_dir, python_exe, ov_source, force):
        if name == "bad_env":
            raise RuntimeError("simulated pip resolution failure")
        venv_path = venvs_dir / name
        (venv_path / "bin").mkdir(parents=True, exist_ok=True)
        (venv_path / "bin" / "python").touch()
        return venv_path

    ran_commands = []

    def fake_run(cmd, env):
        ran_commands.append(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(rme.build_envs, "build_env", fake_build_env)
    monkeypatch.setattr(rme.subprocess, "run", fake_run)

    exit_code = rme.main([
        "--scope", "precommit",
        "--manifest", str(manifest_path),
        "--venvs-dir", str(tmp_path / "venvs"),
        "--results-dir", str(tmp_path / "results"),
    ])

    # bad_env's build failure fails the overall run...
    assert exit_code == 1
    # ...but good_run/good_env still got dispatched despite bad_run/bad_env failing to build.
    assert len(ran_commands) == 1
    assert "good_env" in ran_commands[0][0]


# --- build_envs stamp: a reused venv must match the interpreter and OpenVINO wheels it was built from ---

def _stamp_inputs(tmp_path):
    reqs = tmp_path / "reqs.txt"
    reqs.write_text("pkg==1.0\n")
    wheel = tmp_path / "openvino-1.0-cp311-cp311-manylinux_x86_64.whl"
    wheel.write_bytes(b"wheel-v1")
    return [([str(reqs)], [])], wheel


def test_stamp_changes_when_wheel_content_changes(tmp_path):
    env_steps, wheel = _stamp_inputs(tmp_path)
    before = rme.build_envs.stamp_content(env_steps, "wheels:x", "py311", [wheel])
    wheel.write_bytes(b"wheel-v2")
    assert rme.build_envs.stamp_content(env_steps, "wheels:x", "py311", [wheel]) != before


def test_stamp_changes_when_interpreter_changes(tmp_path):
    env_steps, wheel = _stamp_inputs(tmp_path)
    assert (rme.build_envs.stamp_content(env_steps, "wheels:x", "py311", [wheel])
            != rme.build_envs.stamp_content(env_steps, "wheels:x", "py312", [wheel]))


def test_stamp_is_stable_for_identical_inputs(tmp_path):
    env_steps, wheel = _stamp_inputs(tmp_path)
    assert (rme.build_envs.stamp_content(env_steps, "wheels:x", "py311", [wheel])
            == rme.build_envs.stamp_content(env_steps, "wheels:x", "py311", [wheel]))


def test_resolve_ov_wheels(tmp_path):
    for name in ("openvino-1.0-cp311-cp311-manylinux_x86_64.whl",
                 "openvino-1.0-cp311-cp311t-manylinux_x86_64.whl",
                 "openvino_tokenizers-1.0-py3-none-manylinux_x86_64.whl"):
        (tmp_path / name).write_bytes(b"")
    wheels = rme.build_envs.resolve_ov_wheels(f"wheels:{tmp_path}", "311")
    assert [w.name for w in wheels] == ["openvino-1.0-cp311-cp311-manylinux_x86_64.whl",
                                        "openvino_tokenizers-1.0-py3-none-manylinux_x86_64.whl"]
    assert rme.build_envs.resolve_ov_wheels("nightly", "311") == []
