# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Execute a resumable validation manifest without model calls between cases."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def absolute_path(value, directory=False):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"Expected absolute path: {value!r}")
    if not (Path(value).is_dir() if directory else Path(value).is_file()):
        raise ValueError(f"Missing {'directory' if directory else 'file'}: {value}")


def validate(manifest):
    if manifest.get("version") != 1 or not isinstance(manifest.get("context"), dict) or not manifest["context"]:
        raise ValueError("Expected version 1 and a nonempty context object")
    if not isinstance(manifest.get("repositories"), list) or not manifest["repositories"]:
        raise ValueError("Expected at least one repository")
    for repo in manifest["repositories"]:
        absolute_path(repo, directory=True)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty cases array")
    ids = set()
    report_paths = set()
    for case in cases:
        name = case.get("id", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", name) or name in ids:
            raise ValueError(f"Invalid or duplicate case ID: {name!r}")
        ids.add(name)
        absolute_path(case.get("cwd"), directory=True)
        command = case.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
            raise ValueError(f"{name}: command must be a nonempty argument array")
        timeout = case.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"{name}: timeout_seconds must be positive and finite")
        if not isinstance(case.get("metadata"), dict) or not case["metadata"]:
            raise ValueError(f"{name}: metadata must record the acceptance contract")
        env = case.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError(f"{name}: env must map strings to strings")
        reports = case.get("reports", [])
        if not isinstance(reports, list):
            raise ValueError(f"{name}: reports must be an array")
        for report in reports:
            path = report.get("path")
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError(f"{name}: report path must be absolute")
            resolved = Path(path).resolve()
            if resolved in report_paths:
                raise ValueError(f"{name}: report paths must be unique across cases: {path}")
            report_paths.add(resolved)
            kind = report.get("kind")
            if kind == "gtest":
                count = report.get("expected_tests")
                skips = report.get("max_skips", 0)
                if type(count) is not int or count <= 0 or type(skips) is not int or skips < 0:
                    raise ValueError(f"{name}: gtest requires positive expected_tests and nonnegative max_skips")
            elif kind == "gguf_mmproj":
                fraction = report.get("min_choice_fraction")
                if (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or
                        not math.isfinite(fraction) or not 0 <= fraction <= 1):
                    raise ValueError(f"{name}: gguf_mmproj requires min_choice_fraction in [0, 1]")
                if type(report.get("first_token_matches")) is not bool:
                    raise ValueError(f"{name}: gguf_mmproj requires explicit first_token_matches")
                for key in ("required_modalities", "required_api_checks"):
                    values = report.get(key, [])
                    if (not isinstance(values, list) or not all(isinstance(v, str) and v for v in values) or
                            len(set(values)) != len(values) or (key == "required_modalities" and not values)):
                        raise ValueError(f"{name}: invalid {key}")
                if not isinstance(report.get("equals", {}), dict):
                    raise ValueError(f"{name}: equals must be an object of required top-level report values")
            else:
                raise ValueError(f"{name}: unsupported report kind: {kind!r}")
    for item in [manifest, *cases]:
        if not isinstance(item.get("inputs", []), list):
            raise ValueError("inputs must be an array of files")
        for path in item.get("inputs", []):
            absolute_path(path)
            if Path(path).resolve() in report_paths:
                raise ValueError(f"Report output cannot also be a stable input: {path}")


def repository_state(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", path, *args], stderr=subprocess.PIPE).decode()

    return {
        "head": git("rev-parse", "HEAD").strip(),
        "tracked_diff": digest(git("diff", "HEAD", "--binary", "--no-ext-diff")),
        "submodules": git("submodule", "status", "--recursive").strip(),
    }


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    process.wait()


def audit_report(contract, path):
    if contract["kind"] == "gtest":
        tests = list(ET.parse(path).getroot().iter("testcase"))
        skipped = sum(test.find("skipped") is not None or test.get("status") == "notrun" for test in tests)
        failures = sum(test.find("failure") is not None or test.find("error") is not None for test in tests)
        if len(tests) != contract["expected_tests"] or skipped > contract.get("max_skips", 0) or failures:
            raise ValueError(f"gtest: tests={len(tests)}, skips={skipped}, failures={failures}")
        return {"tests": len(tests), "skips": skipped, "failures": failures}

    report = json.loads(path.read_text())
    if report.get("completed") is not True:
        raise ValueError("GGUF report did not complete")
    for key, value in contract.get("equals", {}).items():
        if key not in report or report[key] != value:
            raise ValueError(f"GGUF report {key} differs from the contract")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("GGUF report has no modality cases")
    modalities = [case["modality"] for case in cases]
    if len(set(modalities)) != len(modalities) or set(contract["required_modalities"]) - set(modalities):
        raise ValueError(f"GGUF modalities missing or duplicated: {modalities}")
    fractions = []
    for case in cases:
        tokens, choices = case["tokens"], case["reference_choices_on_same_history"]
        if (not isinstance(tokens, list) or not isinstance(choices, list) or not tokens or
                len(tokens) != len(choices) or any(type(t) is not int for t in tokens + choices)):
            raise ValueError(f"GGUF {case['modality']}: missing or unequal token histories")
        first = tokens[0] == choices[0]
        fraction = sum(a == b for a, b in zip(tokens, choices)) / len(tokens)
        reported_fraction = case.get("matching_choice_fraction")
        if (case.get("first_token_matches") is not first or isinstance(reported_fraction, bool) or
                not isinstance(reported_fraction, (int, float)) or not math.isfinite(reported_fraction) or
                not math.isclose(reported_fraction, fraction, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"GGUF {case['modality']}: reported scores disagree with token histories")
        if (contract["first_token_matches"] and not first) or fraction < contract["min_choice_fraction"]:
            raise ValueError(f"GGUF {case['modality']}: first_token={first}, choice_fraction={fraction}")
        fractions.append(fraction)
    checks = report.get("api_checks", {})
    if not isinstance(checks, dict) or set(contract.get("required_api_checks", [])) - set(checks):
        raise ValueError("GGUF API checks missing")
    if any(check.get("passed") is not True for check in checks.values()):
        raise ValueError("GGUF API checks failed")
    if report.get("passed") is not True:
        raise ValueError("GGUF report did not pass")
    return {"modalities": modalities, "minimum_choice_fraction": min(fractions), "api_checks": list(checks)}


def execute(case, log):
    started = time.time()
    code = None
    before = {Path(report["path"]): stamp(report["path"]) if Path(report["path"]).is_file() else None
              for report in case.get("reports", [])}
    with log.open("wb") as stream:
        try:
            process = subprocess.Popen(
                case["command"], cwd=case["cwd"], env={**os.environ, **case.get("env", {})},
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
        except OSError as error:
            stream.write(str(error).encode())
            status = "error"
        else:
            try:
                code = process.wait(timeout=case["timeout_seconds"])
                status = "passed" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                stop_process(process)
                status = "timeout"
            except KeyboardInterrupt:
                stop_process(process)
                status = "interrupted"
    result = {"status": status, "returncode": code, "started_at": started,
              "duration_seconds": round(time.time() - started, 3), "log": str(log)}
    artifacts, errors = [], []
    for index, contract in enumerate(case.get("reports", [])):
        path = Path(contract["path"])
        try:
            if not path.is_file() or stamp(path) == before[path]:
                raise ValueError(f"Missing or unchanged report: {path}")
            snapshot = log.with_suffix(f".report-{index}{path.suffix}")
            shutil.copyfile(path, snapshot)
            artifact = {"path": str(snapshot), "sha256": file_digest(snapshot), "source": str(path)}
            artifacts.append(artifact)
            artifact["audit"] = audit_report(contract, snapshot)
        except (ValueError, OSError, KeyError, TypeError, AttributeError, ET.ParseError) as error:
            errors.append(str(error))
    if case.get("reports"):
        result.update({"artifacts": artifacts, "report_errors": errors})
    if errors and status == "passed":
        result["status"] = "failed"
    return result


def run(manifest, output, retry_failed):
    validate(manifest)
    repos = {path: repository_state(path) for path in manifest["repositories"]}
    paths = set(manifest.get("inputs", []))
    for case in manifest["cases"]:
        paths.update(case.get("inputs", []))
    stamps = {path: stamp(path) for path in paths}
    hashes = {path: file_digest(path) for path in sorted(paths)}

    def unchanged():
        return (all(stamp(path) == old for path, old in stamps.items()) and
                all(repository_state(path) == old for path, old in repos.items()))

    if not unchanged():
        raise ValueError("Inputs changed while fingerprinting; use a stable build/worktree")
    provenance = {
        "context": manifest["context"], "repositories": repos,
        "inputs": {path: hashes[path] for path in manifest.get("inputs", [])},
        "environment_hash": digest(dict(os.environ)), "runner_hash": file_digest(__file__),
    }
    records = {}
    fingerprints = {}
    for case in manifest["cases"]:
        name = case["id"]
        fingerprints[name] = digest({"provenance": provenance, "case": case,
                                     "inputs": {path: hashes[path] for path in case.get("inputs", [])}})
        path = output / f"{name}.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        reusable = (old.get("fingerprint") == fingerprints[name] and
                    old.get("status") in {"passed", "failed", "timeout", "error"} and
                    Path(old.get("log", "")).is_file() and
                    all(Path(a["path"]).is_file() and file_digest(a["path"]) == a["sha256"]
                        for a in old.get("artifacts", [])) and
                    not (retry_failed and old.get("status") != "passed"))
        records[name] = old if reusable else {"status": "pending"}

    if (output / "summary.json").exists():
        shutil.copyfile(output / "summary.json", output / f"summary-{time.time_ns()}.json")

    def summary():
        counts = dict(Counter(record["status"] for record in records.values()))
        result = {"counts": counts, "cases": records, "provenance": provenance}
        save(output / "summary.json", result)
        return counts

    summary()
    for case in manifest["cases"]:
        name = case["id"]
        if not unchanged():
            raise ValueError("Inputs changed during validation; stop and prepare a stable build")
        if records[name]["status"] != "pending":
            continue
        records[name] = {"status": "running"}
        summary()
        log = output / f"{name}-{time.time_ns()}.log"
        result = execute(case, log)
        result.update({"fingerprint": fingerprints[name], "case": case, "provenance": provenance,
                       "inputs": {path: hashes[path] for path in case.get("inputs", [])}})
        try:
            stable = unchanged()
        except (OSError, subprocess.CalledProcessError):
            stable = False
        if not stable:
            result["status"] = "stale"
        records[name] = result
        save(log.with_suffix(".json"), result)
        save(output / f"{name}.json", result)
        summary()
        print(f"{name}: {result['status']}; log: {log}", flush=True)
        for error in result.get("report_errors", []):
            print(f"  report: {error}", flush=True)
        if not stable:
            raise ValueError("Inputs changed during the case; result is stale")
        if result["status"] == "interrupted":
            return 130
    counts = summary()
    print(json.dumps(counts), flush=True)
    return 0 if counts.get("passed", 0) == len(records) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another runner owns this output directory") from None
            return run(json.loads(args.manifest.read_text()), output, args.retry_failed)
    except (ValueError, OSError, subprocess.CalledProcessError, TypeError, AttributeError, RuntimeError) as error:
        print(f"Validation setup error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
