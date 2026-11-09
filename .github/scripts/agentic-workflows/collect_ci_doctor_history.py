# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pre-collect the recent CI Doctor knowledge base for the weekly remediation workflow.

Used by the shared pre-agent step
(.github/workflows/shared/agentic-workflows/collect-ci-doctor-history.md). The two
automatic CI Doctors persist their investigations and per-signature failure patterns
on dedicated repo-memory branches:

  - memory/ci-doctor-mq          (subdirectory ``mq/``)          — merge-queue failures
  - memory/ci-doctor-post-commit (subdirectory ``post-commit/``) — post-commit failures

Each branch holds ``<slug>/patterns/<hash>.json`` (one aggregated record per failure
signature) and ``<slug>/investigations/<timestamp>-<run-id>.json`` (one record per
investigation, plus an aggregate ``index.json``). This script downloads those records,
keeps only the ones active within the last ``DAYS`` days, and writes them — together
with a ranked human-readable ``summary.txt`` — under
``/tmp/gh-aw/agent/ci-doctor-remediation/`` so the remediation agent starts from a
compact, pre-digested view instead of walking the memory branches itself.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from typing import TYPE_CHECKING, Any

from common import github_client

if TYPE_CHECKING:
    from github.Repository import Repository

OUTPUT_DIR = "/tmp/gh-aw/agent/ci-doctor-remediation"
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary.txt")

# (repo-memory branch, subdirectory/slug, human label) for each automatic CI Doctor.
DOCTORS = [
    ("memory/ci-doctor-mq", "mq", "Merge Queue"),
    ("memory/ci-doctor-post-commit", "post-commit", "Post-Commit"),
]

# Leading filesystem-safe timestamp of an investigation file name:
# <YYYY-MM-DD-HH-MM-SS-sss>-<run-id>.json
_INVESTIGATION_NAME_TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-\d{3}-")


def _log(message: str) -> None:
    """Print an immediately-flushed progress line (per-file API calls are slow)."""
    print(message, flush=True)


def _cutoff() -> dt.datetime:
    """Return the UTC datetime before which records are considered stale."""
    days = int(os.environ.get("DAYS", "7") or "7")
    return dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)


def _parse_ts(value: str | None) -> dt.datetime | None:
    """Parse an ISO 8601 timestamp (accepting a trailing ``Z``) into an aware datetime."""
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _investigation_ts_from_name(name: str) -> dt.datetime | None:
    """Cheaply recover the UTC timestamp encoded in an investigation file name.

    Lets stale records be skipped before the (per-file) content download.
    """
    match = _INVESTIGATION_NAME_TS_RE.match(name)
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(part) for part in match.groups())
    try:
        return dt.datetime(year, month, day, hour, minute, second, tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _branch_exists(repo: "Repository", ref: str) -> bool:
    """Return whether the ``ref`` branch exists on the repository."""
    try:
        repo.get_branch(ref)
        return True
    except Exception:
        return False


def _list_json(repo: "Repository", ref: str, path: str) -> list[Any]:
    """Return the ``ContentFile`` entries for ``*.json`` files under ``path`` on ``ref``.

    Missing branches or directories are treated as an empty collection so a doctor
    that has not recorded anything yet does not fail the whole run.
    """
    try:
        contents = repo.get_contents(path, ref=ref)
    except Exception as error:
        _log(f"    {ref}:{path} not listable ({error}); treating as empty.")
        return []
    if not isinstance(contents, list):
        contents = [contents]
    entries = [entry for entry in contents if entry.type == "file" and entry.name.endswith(".json")]
    _log(f"    {ref}:{path} -> {len(entries)} json file(s)")
    return entries


def _download_json(entry: Any) -> dict[str, Any] | None:
    """Decode a single JSON ``ContentFile`` entry, returning None on any failure."""
    try:
        return json.loads(entry.decoded_content.decode("utf-8", "replace"))
    except Exception:
        return None


def _pattern_is_recent(record: dict[str, Any], cutoff: dt.datetime) -> bool:
    """A pattern is in-window if it was last seen, or has any recent timestamp, after cutoff."""
    last_seen = _parse_ts(record.get("last_seen"))
    if last_seen is not None and last_seen >= cutoff:
        return True
    return any((ts := _parse_ts(stamp)) is not None and ts >= cutoff for stamp in record.get("recent_timestamps", []))


def _investigation_is_recent(record: dict[str, Any], cutoff: dt.datetime) -> bool:
    """An investigation is in-window if its timestamp is after cutoff."""
    timestamp = _parse_ts(record.get("timestamp"))
    return timestamp is not None and timestamp >= cutoff


def collect_doctor(repo: "Repository", ref: str, slug: str, cutoff: dt.datetime) -> dict[str, Any]:
    """Download and filter one doctor's recent patterns and investigations to disk."""
    patterns_dir = os.path.join(OUTPUT_DIR, slug, "patterns")
    investigations_dir = os.path.join(OUTPUT_DIR, slug, "investigations")
    os.makedirs(patterns_dir, exist_ok=True)
    os.makedirs(investigations_dir, exist_ok=True)

    pattern_entries = _list_json(repo, ref, f"{slug}/patterns")
    patterns: list[tuple[str, dict[str, Any]]] = []
    for index, entry in enumerate(pattern_entries, start=1):
        _log(f"    [{slug}] pattern {index}/{len(pattern_entries)}: downloading {entry.name}")
        record = _download_json(entry)
        if record is None or not _pattern_is_recent(record, cutoff):
            continue
        patterns.append((entry.name, record))

    investigation_entries = [
        entry for entry in _list_json(repo, ref, f"{slug}/investigations") if entry.name != "index.json"
    ]
    # Skip investigations whose file-name timestamp is already out of window before
    # paying for the per-file content download.
    in_window = [
        entry
        for entry in investigation_entries
        if (name_ts := _investigation_ts_from_name(entry.name)) is None or name_ts >= cutoff
    ]
    skipped = len(investigation_entries) - len(in_window)
    _log(f"    [{slug}] investigations: {len(in_window)} in-window by name, {skipped} skipped by name")
    investigation_count = 0
    # Per-signature occurrence tally within the DAYS window. Each in-window
    # investigation is one occurrence carrying the pattern's signature_hash.
    window_counts: dict[str, int] = {}
    for index, entry in enumerate(in_window, start=1):
        _log(f"    [{slug}] investigation {index}/{len(in_window)}: downloading {entry.name}")
        record = _download_json(entry)
        if record is None or not _investigation_is_recent(record, cutoff):
            continue
        investigation_count += 1
        signature_hash = record.get("signature_hash")
        if signature_hash:
            window_counts[signature_hash] = window_counts.get(signature_hash, 0) + 1
        with open(os.path.join(investigations_dir, entry.name), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)

    # Attach a window-local count derived from the in-window investigations. The
    # pattern's own `count` is a lifetime total and `recent_timestamps` is pruned to
    # 24h, so neither reflects the configured DAYS window; `window_count` does.
    ranked: list[dict[str, Any]] = []
    for name, record in patterns:
        record["window_count"] = window_counts.get(record.get("signature_hash"), 0)
        with open(os.path.join(patterns_dir, name), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        ranked.append(record)

    ranked.sort(
        key=lambda record: (record.get("window_count", 0), record.get("count", 0), record.get("last_seen", "")),
        reverse=True,
    )
    return {"patterns": ranked, "investigation_count": investigation_count}


def write_summary(results: dict[str, dict[str, Any]], cutoff: dt.datetime) -> None:
    """Write the ranked, human-readable entry point the agent reads first."""
    with open(SUMMARY_FILE, "w", encoding="utf-8") as handle:
        handle.write("=== CI Doctor Remediation — Pre-Analysis ===\n")
        handle.write(f"Window: records active since {cutoff.isoformat()} (last {os.environ.get('DAYS', '7')} days)\n\n")

        for _ref, slug, label in DOCTORS:
            data = results[slug]
            patterns = data["patterns"]
            handle.write(f"--- {label} doctor ({slug}) ---\n")
            handle.write(f"  Recent investigations: {data['investigation_count']}\n")
            handle.write(f"  Recent failure patterns: {len(patterns)}\n")
            handle.write(f"  Files: {OUTPUT_DIR}/{slug}/patterns/, {OUTPUT_DIR}/{slug}/investigations/\n")
            if patterns:
                handle.write("  Patterns (ranked by in-window occurrence count):\n")
            for record in patterns:
                handle.write(
                    f"    [{record.get('window_count', 0)}x in window, "
                    f"{record.get('count', 0)}x lifetime] {record.get('category', '?')}: "
                    f"{record.get('title', '(no title)')}\n"
                )
                handle.write(
                    f"        signature_hash={record.get('signature_hash', '?')} "
                    f"first_seen={record.get('first_seen', '?')} last_seen={record.get('last_seen', '?')}\n"
                )
                run_urls = record.get("recent_run_urls", [])
                if run_urls:
                    handle.write(f"        recent runs: {', '.join(run_urls[:3])}\n")
                affected_prs = record.get("affected_prs", [])
                if affected_prs:
                    handle.write(f"        affected PRs: {', '.join(affected_prs[:5])}\n")
                affected_commits = record.get("affected_commits", [])
                if affected_commits:
                    handle.write(f"        affected commits: {', '.join(affected_commits[:5])}\n")
            handle.write("\n")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cutoff = _cutoff()
    token = os.environ.get("GH_TOKEN", "")
    repo = github_client(token).get_repo(os.environ.get("REPO", ""))

    _log(f"=== CI Doctor Remediation: collecting knowledge since {cutoff.isoformat()} ===")
    results: dict[str, dict[str, Any]] = {}
    for ref, slug, label in DOCTORS:
        if not _branch_exists(repo, ref):
            _log(f"Skipping {label} doctor: branch {ref} does not exist.")
            results[slug] = {"patterns": [], "investigation_count": 0}
            continue
        _log(f"Collecting {label} doctor from {ref} ({slug}/) ...")
        results[slug] = collect_doctor(repo, ref, slug, cutoff)
        _log(
            f"  -> {len(results[slug]['patterns'])} recent pattern(s), "
            f"{results[slug]['investigation_count']} recent investigation(s)"
        )

    write_summary(results, cutoff)
    _log(f"Summary written to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
