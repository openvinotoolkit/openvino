# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import unittest
import uuid
import warnings
import zipfile

import patch


ROOT = Path(__file__).resolve().parents[3]


def manifest(component, files):
    return (json.dumps({"component": component, "files": files}, indent=2, ensure_ascii=False) + "\n").encode()


def archive(entries):
    data = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as output:
        warnings.simplefilter("ignore", UserWarning)
        for key, value in entries:
            if isinstance(key, str):
                entry = zipfile.ZipInfo()
                entry.filename = entry.orig_filename = key
                entry.compress_type = zipfile.ZIP_DEFLATED
                key = entry
            output.writestr(key, value)
    return base64.b64encode(data.getvalue()).decode()


def gold(files):
    return [{"name": key + ".json", "content": base64.b64encode(value).decode()}
            for key, value in files.items()]


def item(configuration="fixture", actual=None, baseline=None):
    actual = actual if actual is not None else {"core": manifest("core", ["runtime/new.so"])}
    return {"configuration": configuration, "platform": "linux",
            "archive": archive([("linux/" + key + ".json", value) for key, value in actual.items()]),
            "gold": gold(baseline or {})}


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / (".bom-patch-test-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)
        self.git("init", "--quiet")
        self.git("config", "core.autocrlf", "false")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.directory, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def apply(self, platforms):
        for platform in platforms:
            for entry in platform["gold"]:
                path = self.directory / "boms" / platform["configuration"] / entry["name"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(entry["content"]))
        text = patch.make_patch(platforms)
        output = self.directory / "gold.patch"
        output.write_bytes(text.encode("utf-8"))
        self.git("apply", "--check", str(output))
        self.git("apply", str(output))
        self.git("apply", "--reverse", "--check", str(output))
        return text

    def test_exact_patch_modifications_additions_deletions_missing_configuration(self):
        before = b'{"files":["runtime/old.so"],"component":"core"}'  # no final LF
        old = manifest("removed", ["old/path"])
        platforms = [
            item(baseline={"core": before, "removed": old},
                 actual={"core": manifest("core", ["runtime/new.so"]),
                         "added": manifest("added", ['runtime/a "quoted" file.so'])}),
            item("new_configuration"),
        ]
        text = self.apply(platforms)
        self.assertIn("new file mode 100644", text)
        self.assertIn("deleted file mode 100644", text)
        self.assertIn("No newline at end of file", text)
        self.assertEqual((self.directory / "boms/fixture/core.json").read_bytes(),
                         manifest("core", ["runtime/new.so"]))
        self.assertFalse((self.directory / "boms/fixture/removed.json").exists())
        self.assertTrue((self.directory / "boms/new_configuration/core.json").is_file())
        self.assertEqual(text, patch.make_patch(list(reversed(platforms))))

    def test_crlf_unicode_and_missing_final_newlines_apply_exactly(self):
        before = manifest("core", ["old/\u2028name"]).replace(b"\n", b"\r\n")
        after = manifest("core", ["new/\u2028name"]).rstrip(b"\n")
        self.apply([item(actual={"core": after}, baseline={"core": before})])
        self.assertEqual((self.directory / "boms/fixture/core.json").read_bytes(), after)

    def test_semantically_equal_manifests_do_not_reformat_gold(self):
        data = item(actual={"core": manifest("core", ["a", "b"])},
                    baseline={"core": b'{"component":"core","files":["b","a"]}'})
        self.assertEqual(patch.make_patch([data]), "")

    def test_patch_is_tied_to_exact_pr_gold_bytes(self):
        data = item(baseline={"core": manifest("core", ["old"])})
        text = self.apply([data])
        output = self.directory / "gold.patch"
        output.write_text(text, encoding="utf-8")
        result = subprocess.run(["git", "apply", "--check", str(output)],
                                cwd=self.directory, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_archive_entries(self):
        for filename in ("../core.json", "/linux/core.json", "linux/../core.json",
                         "linux/nested/core.json", "linux\\core.json", "windows/core.json",
                         "linux/core.txt", "linux/-bad.json", "linux/CON.json",
                         "linux/core.json\x00hidden"):
            with self.subTest(filename=filename):
                data = archive([(filename, manifest("core", ["file"]))])
                with self.assertRaises((ValueError, zipfile.BadZipFile)):
                    patch.generated(data, "linux")

    def test_duplicate_case_collision_and_symlink_archive_entries(self):
        for entries in [
            [("linux/core.json", manifest("core", []))] * 2,
            [("linux/core.json", manifest("core", [])), ("linux/Core.json", manifest("Core", []))],
            [],
        ]:
            with self.assertRaises(ValueError):
                patch.generated(archive(entries), "linux")
        link = zipfile.ZipInfo("linux/core.json")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError):
            patch.generated(archive([(link, b"target")]), "linux")

    def test_archive_file_and_expansion_limits(self):
        data = item()
        original = patch.MAX_FILE
        try:
            patch.MAX_FILE = 8
            with self.assertRaisesRegex(ValueError, "expansion limit"):
                patch.make_patch([data])
        finally:
            patch.MAX_FILE = original

    def test_invalid_manifests_never_generate_patch_including_component_changes(self):
        invalid = [
            b'{"component":"wrong","files":[]}',
            b'{"component":"core","files":[],"extra":1}',
            b'{"component":"core","component":"core","files":[]}',
            b'{"component":"core","files":["a","a"]}',
            b'{"component":"core","files":"file"}',
            b'{"component":"core","files":[1]}', b"not-json", b"\xff",
        ]
        for file in ("../file", "/absolute", "a//b", "a/./b", "C:/file", "a\\b", "a\nb"):
            invalid.append(manifest("core", [file]))
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises((ValueError, UnicodeError)):
                    patch.make_patch([item(actual={"core": raw}, baseline={"removed": manifest("removed", [])})])
                with self.assertRaises((ValueError, UnicodeError)):
                    patch.make_patch([item(baseline={"core": raw})])

    def test_unsafe_configuration_or_gold_name(self):
        for configuration in ("../escape", "-option", "space name", "bad\nname", "CON", "a."):
            with self.assertRaises(ValueError):
                patch.make_patch([item(configuration)])
        data = item()
        data["gold"] = [{"name": "../evil.json", "content": base64.b64encode(b"{}").decode()}]
        with self.assertRaises(ValueError):
            patch.make_patch([data])


class PolicyTests(unittest.TestCase):
    def workflows(self):
        # Read the production JS mapping without introducing another copy.
        output = subprocess.check_output(
            ["node", "-e", "console.log(JSON.stringify(require('./.github/scripts/bom_comments/comment.js').PLATFORMS))"],
            cwd=ROOT, text=True)
        mapping = json.loads(output)
        return [{"file": file, "name": value[0], "configuration": value[1],
                 "head": (ROOT / ".github/workflows" / file).read_text(encoding="utf-8"),
                 "trusted": (ROOT / ".github/workflows" / file).read_text(encoding="utf-8")}
                for file, value in mapping.items()]

    def test_real_pr_triggers_and_path_filtered_windows(self):
        workflows = self.workflows()
        expected = patch.policy(workflows, ["src/core/file.cpp"], "master")
        files = {value["file"] for value in expected}
        self.assertEqual(len(files), 7)
        self.assertNotIn("mac_arm64.yml", files)
        self.assertNotIn("ubuntu_26.yml", files)
        self.assertNotIn("windows_vs2026_release.yml", files)
        selected = patch.policy(workflows, [".github/workflows/windows_vs2026_release.yml"], "releases/2026")
        self.assertEqual(len(selected), 8)
        # GitHub's 300-file path filter boundary cannot shrink the expected set.
        self.assertEqual(len(patch.policy(workflows, ["src/file"] * 301, "master")), 8)

    def test_trigger_or_build_gating_changes_fail_closed(self):
        for before, after in [
            ("  pull_request:", "  disabled_pull_request:"),
            ('if: "!needs.smart_ci.outputs.skip_workflow"', 'if: "false"'),
            ("needs: [Docker, Smart_CI]", "needs: [Smart_CI]"),
        ]:
            workflows = self.workflows()
            target = next(value for value in workflows if value["file"] == "ubuntu_22.yml")
            self.assertIn(before, target["head"])
            target["head"] = target["head"].replace(before, after)
            with self.assertRaises(ValueError):
                patch.policy(workflows, ["src/file"], "master")

    def test_yaml_duplicate_keys_aliases_and_malformed_input_fail_closed(self):
        for text in ("on: {}\non: {}", "on: &alias {}\nother: *alias", "on: ["):
            with self.assertRaises(Exception):
                patch.workflow(text)


if __name__ == "__main__":
    unittest.main()
