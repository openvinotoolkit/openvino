# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Audit a recorded compiler link command without executing it."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import sys


OBJECT_SUFFIXES = (".o", ".obj")
MAX_RESPONSE_DEPTH = 128
OUTPUT_OPTIONS = {"-o", "--output"}
PATH_OPTIONS = {
    "-D", "-I", "-L", "-MF", "-MT", "-MQ", "-T", "-U",
    "--Map", "--dynamic-linker", "--entry", "--output", "--rpath",
    "--rpath-link", "--script", "--soname", "--sysroot", "--undefined",
    "-dynamic-linker", "-e", "-include", "-imacros", "-isystem",
    "-isysroot", "-iquote", "-idirafter", "-Map", "-rpath", "-rpath-link",
    "-soname", "-u", "-z",
}


class AuditInputError(ValueError):
    pass


def canonical_path(value, directory):
    path = Path(value)
    if not path.is_absolute():
        path = directory / path
    return path.resolve()


def aliases(first, second):
    if first.resolve() == second.resolve():
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def parse_command(command):
    tokens = []
    token = []
    started = False
    quote = None
    index = 0
    operators = ";&|<>()"
    while index < len(command):
        char = command[index]
        if quote == "'":
            if char == "'":
                quote = None
            else:
                token.append(char)
            index += 1
            continue
        if char == "\\":
            if index + 1 == len(command):
                raise AuditInputError("cannot parse command text: trailing escape")
            escaped = command[index + 1]
            if quote == '"' and escaped not in '$`"\\\n':
                token.append(char)
                index += 1
                continue
            if escaped != "\n":
                token.append(escaped)
                started = True
            index += 2
            continue
        if quote == '"':
            if char == '"':
                quote = None
            elif char == "`" or (char == "$" and index + 1 < len(command) and command[index + 1] == "("):
                raise AuditInputError("command substitution is not allowed")
            elif char == "$":
                raise AuditInputError("shell parameter expansion is not allowed; record literal arguments")
            else:
                token.append(char)
            index += 1
            continue
        if char in "'\"":
            quote = char
            started = True
        elif char == "#" and not started:
            newline = command.find("\n", index)
            if newline < 0:
                break
            index = newline
            continue
        elif char == "`" or (char == "$" and index + 1 < len(command) and command[index + 1] == "("):
            raise AuditInputError("command substitution is not allowed")
        elif char == "$":
            raise AuditInputError("shell parameter expansion is not allowed; record literal arguments")
        elif char in "*?[{}~":
            raise AuditInputError(
                "shell pathname, brace and tilde expansions are not allowed; record literal arguments"
            )
        elif char in operators:
            raise AuditInputError("shell operators and multiple commands are not allowed")
        elif char == "\n" and command[index + 1 :].strip():
            raise AuditInputError("multiple shell commands are not allowed")
        elif char in " \t\n":
            if started:
                tokens.append("".join(token))
                token = []
                started = False
        else:
            token.append(char)
            started = True
        index += 1
    if quote:
        raise AuditInputError("cannot parse command text: unterminated quote")
    if started:
        tokens.append("".join(token))
    if not tokens:
        raise AuditInputError("command text is empty")
    return tokens


def normalize_linker_tokens(tokens):
    normalized = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-Xlinker":
            if index + 1 >= len(tokens):
                raise AuditInputError("-Xlinker is missing its argument")
            normalized.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("-Wl,"):
            normalized.extend(part for part in token[4:].split(",") if part)
        else:
            normalized.append(token)
        index += 1
    return normalized


def load_contract(path, directory):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditInputError(f"cannot read contract JSON: {error}")
    if not isinstance(data, dict):
        raise AuditInputError("contract must be a JSON object")
    if "required_objects" not in data:
        raise AuditInputError("contract is missing required_objects")
    required = data["required_objects"]
    forbidden = data.get("forbidden_objects")
    if not isinstance(required, list) or not required:
        raise AuditInputError("required_objects must be a nonempty list of object paths")
    if not isinstance(forbidden, list):
        raise AuditInputError("forbidden_objects must be a list of object paths")

    result = {}
    for key, values in (("required_objects", required), ("forbidden_objects", forbidden)):
        paths = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise AuditInputError(f"{key} entries must be nonempty path strings")
            path_value = canonical_path(value, directory)
            if path_value.suffix not in OBJECT_SUFFIXES:
                raise AuditInputError(f"{key} entry is not an .o/.obj path: {value}")
            paths.append(str(path_value))
        if len(paths) != len(set(paths)):
            raise AuditInputError(f"{key} contains duplicate paths")
        result[key] = paths
    overlap = sorted(set(result["required_objects"]) & set(result["forbidden_objects"]))
    if overlap:
        raise AuditInputError(f"objects cannot be both required and forbidden: {overlap[0]}")
    return result


def expand_responses(tokens, directory, responses, stack=()):
    expanded = []
    for token in normalize_linker_tokens(tokens):
        if not token.startswith("@"):
            expanded.append(token)
            continue
        name = token[1:]
        if not name:
            raise AuditInputError("response-file reference has an empty path")
        response = canonical_path(name, directory)
        responses.add(response)
        if response in stack:
            chain = " -> ".join(str(item) for item in stack + (response,))
            raise AuditInputError(f"response-file cycle: {chain}")
        if len(stack) >= MAX_RESPONSE_DEPTH:
            raise AuditInputError(f"response-file nesting exceeds {MAX_RESPONSE_DEPTH} levels")
        try:
            contents = response.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise AuditInputError(f"cannot read response file {response}: {error}")
        try:
            nested = shlex.split(contents, comments=False, posix=True)
        except ValueError as error:
            raise AuditInputError(f"cannot parse response file {response}: {error}")
        expanded.extend(expand_responses(nested, directory, responses, stack + (response,)))
    return expanded


def object_occurrences(tokens, directory):
    objects = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in PATH_OPTIONS:
            skip_next = True
            continue
        if token in OUTPUT_OPTIONS:
            skip_next = True
            continue
        if token.startswith("-") or not token.endswith(OBJECT_SUFFIXES):
            continue
        objects.append(str(canonical_path(token, directory)))
    return Counter(objects)


def audit(command_file, contract_file, directory, output_file):
    report = {
        "passed": False,
        "cwd": str(directory),
        "command_file": str(command_file.resolve()),
        "contract_file": str(contract_file.resolve()),
        "output": str(output_file.resolve()),
        "counts": {"command_tokens": 0, "expanded_tokens": 0, "response_files": 0,
                   "object_occurrences": 0, "unique_objects": 0},
        "objects": [],
        "response_files": [],
        "violations": [],
        "parse_errors": [],
        "scope": (
            "Checks only recorded object paths; does not establish source/header dependencies "
            "or runtime loaded identity."
        ),
    }
    inputs = {command_file.resolve(), contract_file.resolve()}
    responses = set()
    response_discovery_complete = False

    try:
        if not directory.is_dir():
            raise AuditInputError(f"working directory is not a directory: {directory}")
        raw_command = command_file.read_text(encoding="utf-8")
        command_tokens = parse_command(raw_command)
        report["counts"]["command_tokens"] = len(command_tokens)
        expanded = expand_responses(command_tokens, directory, responses)
        response_discovery_complete = True
        report["counts"]["expanded_tokens"] = len(expanded)
        report["counts"]["response_files"] = len(responses)
        report["response_files"] = sorted(str(path) for path in responses)
        counts = object_occurrences(expanded, directory)
        inputs.update(Path(path) for path in counts)
        report["counts"]["object_occurrences"] = sum(counts.values())
        report["counts"]["unique_objects"] = len(counts)
        report["objects"] = [{"path": path, "count": count} for path, count in sorted(counts.items())]
        contract = load_contract(contract_file, directory)

        for path, count in sorted(counts.items()):
            if count > 1:
                report["violations"].append({
                    "kind": "duplicate_object",
                    "path": path,
                    "count": count,
                    "message": f"object appears {count} times: {path}",
                })
        for path in contract["required_objects"]:
            count = counts[path]
            if count != 1:
                report["violations"].append({
                    "kind": "required_object_count",
                    "path": path,
                    "expected": 1,
                    "actual": count,
                    "message": f"required object appears {count} times; expected exactly once: {path}",
                })
        for path in contract["forbidden_objects"]:
            count = counts[path]
            if count:
                report["violations"].append({
                    "kind": "forbidden_object",
                    "path": path,
                    "actual": count,
                    "message": f"forbidden object appears {count} times: {path}",
                })
    except (ValueError, OSError, UnicodeError, RuntimeError) as error:
        report["parse_errors"].append(str(error))
    inputs.update(responses)
    report["counts"]["response_files"] = len(responses)
    report["response_files"] = sorted(str(path) for path in responses)

    report["passed"] = not report["violations"] and not report["parse_errors"]
    return report, inputs, response_discovery_complete


def write_report(requested, report, inputs):
    destination = requested.resolve()
    collision = any(aliases(destination, path) for path in inputs)
    if collision:
        report["parse_errors"].append("output path aliases an input file")
        report["passed"] = False
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    report["output"] = str(destination)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-file", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    directory = args.directory.resolve()
    command_file = args.command_file.resolve()
    contract_file = args.contract.resolve()
    output_file = args.output.resolve()
    report, inputs, response_discovery_complete = audit(
        command_file, contract_file, directory, output_file
    )
    destination = (
        write_report(output_file, report, inputs)
        if response_discovery_complete else None
    )
    first_violations = report["violations"][:5]
    remaining = max(0, 5 - len(first_violations))
    first_violations.extend(
        {"kind": "parse_error", "message": issue}
        for issue in report["parse_errors"][:remaining]
    )
    summary = {
        "passed": report["passed"],
        "counts": report["counts"],
        "violations": len(report["violations"]),
        "parse_errors": len(report["parse_errors"]),
        "first_violations": first_violations,
        "omitted_violations": max(0, len(report["violations"]) + len(report["parse_errors"]) - 5),
        "details": str(destination) if destination else None,
    }
    if destination is None:
        summary["output_refused"] = str(output_file)
        if not response_discovery_complete:
            summary["output_refusal_reason"] = "response-file discovery was incomplete"
    print(json.dumps(summary, sort_keys=True))
    if report["parse_errors"]:
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
