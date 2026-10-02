# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import validate


class ValidateSkillsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name)
        self.skill = self.repository / ".claude/skills/example"
        self.skill.mkdir(parents=True)
        (self.repository / ".agents").mkdir()
        (self.repository / ".agents/skills").symlink_to("../.claude/skills", target_is_directory=True)
        self.write_skill()

    def write_skill(self, metadata=None, body="Review the proposed change."):
        if metadata is None:
            metadata = {"name": "example", "description": "Review a proposed change."}
        (self.skill / "SKILL.md").write_text(
            "---\n" + validate.yaml.safe_dump(metadata) + "---\n" + body, encoding="utf-8"
        )

    def test_valid_optional_metadata(self):
        self.write_skill({
            "name": "example", "description": "Review changes.", "license": "Apache-2.0",
            "compatibility": "Requires Python 3.10", "metadata": {"version": "1.0"},
            "allowed-tools": "Read",
        })
        self.assertEqual(validate.validate(self.repository), (1, []))

    def test_invalid_metadata(self):
        for field, value in (
            ("name", "Different"), ("name", "example--skill"), ("name", "a" * 65),
            ("description", ""), ("description", 42), ("description", "x" * 1025),
            ("compatibility", "x" * 501), ("metadata", {"version": 1}),
            ("allowed-tools", ["Read"]), ("unexpected", True),
        ):
            with self.subTest(field=field, value=value):
                metadata = {"name": "example", "description": "Review changes."}
                metadata[field] = value
                self.write_skill(metadata)
                self.assertTrue(validate.validate(self.repository)[1])

    def test_malformed_frontmatter_and_empty_body(self):
        for text in (
            "No frontmatter", "---\nname: example", "---\n[]\n---\nBody",
            "---\nname: [\n---\nBody",
            "---\nname: example\nname: duplicate\n---\nBody",
            "---\nname: example\ndescription: Review changes.\n---\n",
        ):
            with self.subTest(text=text):
                (self.skill / "SKILL.md").write_text(text, encoding="utf-8")
                self.assertTrue(validate.validate(self.repository)[1])

    def test_missing_entrypoint_fails_but_removing_whole_skill_passes(self):
        (self.skill / "SKILL.md").unlink()
        self.assertIn("missing SKILL.md", validate.validate(self.repository)[1][0])
        self.skill.rmdir()
        self.assertEqual(validate.validate(self.repository), (0, []))

    def test_cli_exit_status(self):
        command = [sys.executable, validate.__file__, "--repo-root", str(self.repository)]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.write_skill(body="")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("SKILL.md: missing skill instructions", result.stdout)

    def test_entrypoint_misplaced_at_collection_root(self):
        (self.skill / "SKILL.md").rename(self.skill.parent / "SKILL.md")
        self.skill.rmdir()
        self.assertIn("each skill must have its own directory", validate.validate(self.repository)[1][0])


if __name__ == "__main__":
    unittest.main()
