# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data-only policy validation and deterministic, bounded GOLD patch generation."""

import base64
import io
import json
import re
import stat
import sys
import zipfile


MAX_FILE = 2 * 1024 * 1024
MAX_TOTAL = 32 * 1024 * 1024
MAX_COMPONENTS = 512
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def name(value):
    require(isinstance(value, str) and NAME.fullmatch(value), "Unsafe BOM name")
    require(value.split(".")[0].upper() not in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
        *(f"LPT{i}" for i in range(10)),
    } and not value.endswith("."), "Non-portable BOM name")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def manifest(raw, component):
    require(len(raw) <= MAX_FILE, "BOM manifest is too large")
    text = raw.decode("utf-8")
    value = json.loads(text, object_pairs_hook=unique_object)
    require(isinstance(value, dict) and set(value) == {"component", "files"} and
            value["component"] == component, "Invalid BOM component/schema")
    paths = value["files"]
    require(isinstance(paths, list) and len(paths) <= 100000, "Invalid BOM files")
    for item in paths:
        require(isinstance(item, str) and 0 < len(item) <= 4096 and
                not re.search(r"[\x00-\x1f\x7f-\x9f\\:]", item) and
                not any(0xD800 <= ord(char) <= 0xDFFF for char in item) and
                all(part not in {"", ".", ".."} for part in item.split("/")),
                "Unsafe packaged file path")
    require(len(paths) == len(set(paths)), "Duplicate packaged file path")
    return text, set(paths)


def generated(encoded, platform):
    raw = base64.b64decode(encoded, validate=True)
    require(len(raw) <= MAX_TOTAL, "BOM archive is too large")
    result = {}
    total = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= MAX_COMPONENTS + 1, "Invalid BOM archive entry count")
        seen = set()
        for entry in entries:
            require(entry.filename not in seen and entry.filename == entry.orig_filename,
                    "Duplicate or malformed archive entry")
            seen.add(entry.filename)
            mode = entry.external_attr >> 16
            require(not entry.flag_bits & 1 and entry.compress_type in {
                zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
            }, "Unsupported BOM ZIP encoding")
            if entry.filename == platform + "/":
                require(entry.is_dir() and entry.file_size == 0 and
                        stat.S_IFMT(mode) in {0, stat.S_IFDIR}, "Invalid platform directory")
                continue
            require(stat.S_IFMT(mode) in {0, stat.S_IFREG} and not entry.is_dir(),
                    "Non-regular BOM archive entry")
            parts = entry.filename.split("/")
            require(len(parts) == 2 and parts[0] == platform and parts[1].endswith(".json"),
                    "Unexpected BOM archive path")
            component = name(parts[1][:-5])
            require(component.casefold() not in {key.casefold() for key in result},
                    "Case-colliding BOM components")
            total += entry.file_size
            require(0 < entry.file_size <= MAX_FILE and total <= MAX_TOTAL,
                    "BOM archive expansion limit exceeded")
            data = archive.read(entry)
            manifest(data, component)
            result[component] = data
    require(0 < len(result) <= MAX_COMPONENTS, "BOM archive contains no manifests or too many components")
    return result


def baseline(files):
    require(len(files) <= MAX_COMPONENTS, "Too many GOLD manifests")
    result = {}
    total = 0
    for item in files:
        component = name(item["name"][:-5])
        require(item["name"] == component + ".json" and
                component.casefold() not in {key.casefold() for key in result},
                "Invalid or duplicate GOLD name")
        raw = base64.b64decode(item["content"], validate=True)
        total += len(raw)
        require(total <= MAX_TOTAL, "GOLD size limit exceeded")
        manifest(raw, component)
        result[component] = raw
    return result


def lines(raw, prefix):
    text = raw.decode("utf-8")
    # Split only on LF: preserve CRLF and exact bytes from the PR's GOLD blobs.
    result = []
    parts = text.split("\n")
    for line in parts[:-1]:
        result.append(prefix + line + "\n")
    if parts[-1]:
        result.append(prefix + parts[-1] + "\n\\ No newline at end of file\n")
    return result


def make_patch(platforms):
    patch = []
    configurations = set()
    for item in sorted(platforms, key=lambda value: value["configuration"]):
        configuration = name(item["configuration"])
        require(configuration not in configurations, "Duplicate configuration")
        configurations.add(configuration)
        require(item["platform"] in {"linux", "windows", "macos"}, "Unsupported BOM platform")
        actual = generated(item["archive"], item["platform"])
        gold = baseline(item["gold"])
        for component in sorted(set(actual) | set(gold)):
            before, after = gold.get(component), actual.get(component)
            if before is not None and after is not None:
                if manifest(before, component)[1] == manifest(after, component)[1]:
                    continue
            path = f"boms/{configuration}/{component}.json"
            patch.append(f"diff --git a/{path} b/{path}\n")
            if before is None:
                patch.append("new file mode 100644\n")
            elif after is None:
                patch.append("deleted file mode 100644\n")
            patch.extend([
                f"--- {'a/' + path if before is not None else '/dev/null'}\n",
                f"+++ {'b/' + path if after is not None else '/dev/null'}\n",
            ])
            old = lines(before or b"", "-")
            new = lines(after or b"", "+")
            old_count = len((before or b"").split(b"\n")) - int((before or b"").endswith(b"\n"))
            new_count = len((after or b"").split(b"\n")) - int((after or b"").endswith(b"\n"))
            if before is None:
                old_count = 0
            if after is None:
                new_count = 0
            patch.append(f"@@ -{1 if old_count else 0},{old_count} +{1 if new_count else 0},{new_count} @@\n")
            patch.extend(old + new)
    text = "".join(patch)
    require(len(text.encode("utf-8")) <= 64 * 1024 * 1024, "Combined patch size limit exceeded")
    return text


def workflow(text):
    import yaml

    require(len(text.encode("utf-8")) <= 256 * 1024, "Workflow is too large")

    class Loader(yaml.BaseLoader):
        def compose_node(self, parent, index):
            require(not self.check_event(yaml.AliasEvent), "YAML aliases are not allowed")
            self.node_count = getattr(self, "node_count", 0) + 1
            require(self.node_count <= 20000, "Workflow node limit exceeded")
            return super().compose_node(parent, index)

        def construct_mapping(self, node, deep=False):
            return unique_object((self.construct_object(key, deep=deep),
                                  self.construct_object(value, deep=deep))
                                 for key, value in node.value)

    value = yaml.load(text, Loader=Loader)
    require(isinstance(value, dict) and isinstance(value.get("on"), dict),
            "Unsupported workflow triggers")
    return value


def policy(items, changed_files, base_ref, large_diff=False):
    expected = []
    for item in items:
        trusted, head = workflow(item["trusted"]), workflow(item["head"])
        require(trusted.get("name") == head.get("name") == item["name"],
                "Workflow name differs from trusted policy")
        require("pull_request_target" not in head["on"], "Elevated PR trigger is not allowed")
        trigger = trusted["on"].get("pull_request")
        require(("pull_request" in trusted["on"]) == ("pull_request" in head["on"]) and
                trigger == head["on"].get("pull_request"),
                "PR trigger differs from trusted policy; CI maintainer review required")
        if "pull_request" not in trusted["on"]:
            continue
        trigger = trigger or {}
        require(isinstance(trigger, dict) and
                set(trigger) <= {"types", "paths", "branches", "paths-ignore", "branches-ignore"},
                "Unsupported PR trigger policy")
        require(trigger.get("types", ["opened", "synchronize", "reopened"]) ==
                ["opened", "synchronize", "reopened", "ready_for_review"],
                "Unsupported PR activity policy")

        def matches(pattern, value):
            require(isinstance(pattern, str) and len(pattern) <= 512 and
                    not re.search(r"[\[\]{}!+\\]", pattern), "Unsupported trigger glob")
            expression = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
            return re.fullmatch(expression, value) is not None

        def selected(include, exclude, values):
            require(not (include and exclude), "Conflicting trigger filters")
            for patterns in (include, exclude):
                require(patterns is None or isinstance(patterns, list), "Invalid trigger filters")
            return ((include is None or any(matches(p, v) for p in include for v in values)) and
                    (exclude is None or any(not any(matches(p, v) for p in exclude) for v in values)))

        if not selected(trigger.get("branches"), trigger.get("branches-ignore"), [base_ref]):
            continue
        # GitHub only evaluates the first 300 files. Include conservatively beyond
        # that boundary instead of falsely deciding a workflow cannot be expected.
        if not large_diff and len(changed_files) <= 300 and not selected(
                trigger.get("paths"), trigger.get("paths-ignore"), changed_files):
            continue
        jobs = trusted["jobs"]
        head_jobs = head["jobs"]
        for key in ("Build", "Smart_CI", "BOM_Tests"):
            require(key in jobs and key in head_jobs, "Missing product job")
            for field in ("if", "needs", "uses", "name"):
                require(jobs[key].get(field) == head_jobs[key].get(field),
                        "Product job gating differs from trusted policy")
        require(jobs["Build"].get("if") == "!needs.smart_ci.outputs.skip_workflow",
                "Unsupported Build gating")
        needs = jobs["Build"]["needs"]
        require(needs in (["Smart_CI"], ["Docker", "Smart_CI"]), "Unsupported Build dependencies")
        expected.append({"file": item["file"], "configuration": item["configuration"],
                         "name": item["name"], "dependencies": needs})
    return expected


def main():
    request = json.load(sys.stdin, object_pairs_hook=unique_object)
    if request["operation"] == "policy":
        result = policy(request["workflows"], request["files"], request["base"], request.get("largeDiff", False))
    else:
        require(request["operation"] == "patch", "Unknown operation")
        result = make_patch(request["platforms"])
    json.dump(result, sys.stdout, ensure_ascii=True)


if __name__ == "__main__":
    main()
