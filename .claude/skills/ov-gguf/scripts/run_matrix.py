# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run commands in manifest order; audit GTest XML or GGUF GenAI JSON reports."""

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
from xml.etree.ElementTree import ParseError

import matrix_reports


# Input identity and checkpoint helpers


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


# Manifest validation


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
                matrix_reports.validate_gtest_contract(name, report)
            elif kind == "gguf_mmproj":
                matrix_reports.validate_gguf_mmproj_contract(name, report)
            else:
                raise ValueError(f"{name}: unsupported report kind: {kind!r}")
    for item in [manifest, *cases]:
        if not isinstance(item.get("inputs", []), list):
            raise ValueError("inputs must be an array of files")
        for path in item.get("inputs", []):
            absolute_path(path)
            if Path(path).resolve() in report_paths:
                raise ValueError(f"Report output cannot also be a stable input: {path}")


# Command execution


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
    collect_reports(case, log, before, result)
    return result


def collect_reports(case, log, before, result):
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
            artifact["audit"] = matrix_reports.audit_report(contract, snapshot)
        except (ValueError, OSError, KeyError, TypeError, AttributeError, ParseError) as error:
            errors.append(str(error))
    if case.get("reports"):
        result.update({"artifacts": artifacts, "report_errors": errors})
    if errors and result["status"] == "passed":
        result["status"] = "failed"


# Resume and batch execution


def can_reuse(record, fingerprint, retry_failed):
    if record.get("fingerprint") != fingerprint:
        return False
    if record.get("status") not in {"passed", "failed", "timeout", "error"}:
        return False
    if retry_failed and record["status"] != "passed":
        return False
    if not Path(record.get("log", "")).is_file():
        return False
    return all(Path(a["path"]).is_file() and file_digest(a["path"]) == a["sha256"]
               for a in record.get("artifacts", []))


def run(manifest, output, retry_failed):
    validate(manifest)
    repos = {path: repository_state(path) for path in manifest["repositories"]}
    runner_paths = [str(Path(__file__).resolve()), str(Path(matrix_reports.__file__).resolve())]
    paths = set(manifest.get("inputs", []))
    paths.update(runner_paths)
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
        "environment_hash": digest(dict(os.environ)),
        "runner_hash": digest({path: hashes[path] for path in runner_paths}),
    }
    records = {}
    fingerprints = {}
    for case in manifest["cases"]:
        name = case["id"]
        fingerprints[name] = digest({"provenance": provenance, "case": case,
                                     "inputs": {path: hashes[path] for path in case.get("inputs", [])}})
        path = output / f"{name}.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        records[name] = old if can_reuse(old, fingerprints[name], retry_failed) else {"status": "pending"}

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


# Command-line entrypoint


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
