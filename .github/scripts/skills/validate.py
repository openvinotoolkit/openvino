# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validate .claude/skills metadata and structure without executing skills."""

import argparse
from pathlib import Path

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently overwriting metadata."""

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in mapping:
                raise yaml.constructor.ConstructorError(
                    None, None, "mapping keys must be unique strings", key_node.start_mark
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def split_frontmatter(text):
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter (---)")
    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise ValueError("missing closing --- for YAML frontmatter") from error
    return yaml.load("\n".join(lines[1:end]), Loader=UniqueKeyLoader), "\n".join(lines[end + 1:])


def metadata_errors(metadata, directory_name):
    """Check fields defined by https://agentskills.io/specification."""
    if not isinstance(metadata, dict):
        return ["frontmatter must be a mapping"]

    errors = []
    fields = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
    for field in sorted(metadata.keys() - fields):
        errors.append(f"unknown frontmatter field: {field}")

    name = metadata.get("name")
    if not isinstance(name, str) or not 1 <= len(name) <= 64:
        errors.append("name must be a string of 1-64 characters")
    else:
        valid_characters = all((char.isalnum() and char == char.lower()) or char == "-" for char in name)
        if not valid_characters or name.startswith("-") or name.endswith("-") or "--" in name:
            errors.append("name must contain lowercase letters, numbers, and single internal hyphens")
        if name != directory_name:
            errors.append(f"name must match skill directory '{directory_name}'")

    for field, limit in (("description", 1024), ("compatibility", 500)):
        if field == "compatibility" and field not in metadata:
            continue
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            errors.append(f"{field} must be a non-empty string of at most {limit} characters")

    for field in ("license", "allowed-tools"):
        if field in metadata and not isinstance(metadata[field], str):
            errors.append(f"{field} must be a string")
    if "metadata" in metadata:
        value = metadata["metadata"]
        if not isinstance(value, dict) or any(
                not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
            errors.append("metadata must map string keys to string values")
    return errors


def validate(repository):
    repository = repository.resolve()
    root = repository / ".claude/skills"
    errors = []
    count = 0
    # Removing the last skill is a valid change.
    if not root.exists():
        return count, errors
    if (root / "SKILL.md").exists():
        errors.append(".claude/skills/SKILL.md: each skill must have its own directory")

    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        skill = directory / "SKILL.md"
        if not skill.is_file():
            errors.append(f"{skill.relative_to(repository)}: missing SKILL.md")
        else:
            count += 1

        for source in sorted(directory.rglob("SKILL.md")):
            relative = source.relative_to(repository)
            try:
                text = source.read_text(encoding="utf-8")
                metadata, text = split_frontmatter(text)
                errors.extend(f"{relative}: {error}" for error in metadata_errors(metadata, source.parent.name))
                if not text.strip():
                    errors.append(f"{relative}: missing skill instructions")
            except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
                errors.append(f"{relative}: {error}")
    return count, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    if not args.repo_root.is_dir():
        parser.error("--repo-root must be an existing directory")
    count, errors = validate(args.repo_root)
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Checked {count} skill(s); {len(errors)} error(s).")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
