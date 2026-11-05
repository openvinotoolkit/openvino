# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Propose a new workflow-rerunner search string for human review.

Custom safe-output job for the CI Doctor MQ workflow
(.github/workflows/shared/agentic-workflows/propose-rerun-pattern.md).

The workflow rerunner (.github/scripts/workflow_rerun) only reads
``errors_to_look_for.json`` from the default branch. The CI Doctor agent never
edits it: it calls ``propose_rerun_pattern`` with a candidate; this job validates
the candidate deterministically and, if it passes, opens a pull request against
the default branch that appends one entry to ``errors_to_look_for.json``. The
string becomes active only after a maintainer merges the PR.

A rejected or duplicate candidate is not an error: the job logs the reason (also
to the step summary) and exits successfully.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from common import github_client, read_agent_item, require_env, resolve_repository

TRANSIENT_CATEGORIES = ("Flaky Test", "Infrastructure", "Network", "External Service")
SIGNATURE_HASH_RE = re.compile(r"[0-9a-f]{8,64}")
BRANCH_PREFIX = "ci-doctor/rerun-pattern/"
TITLE_PREFIX = "[ci-doctor] "

MIN_WORDS = 3
MIN_CHARS = 15

# "##[group]Run ..." / "##[endgroup]" markers delimit steps in every job log.
STEP_HEADER_RE = re.compile(r"^\s*(\S+\s+)?##\[(group|endgroup)\]")

# A URL (and the versioned file name it carries) ties a string to one dependency release,
# so it never matches the same failure for another URL or after a version bump.
URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S")

# Words that, on their own, appear in almost every failing (and many passing) log.
# A candidate made up only of these (and numbers) is too generic to auto-rerun on.
GENERIC_WORDS = frozenset(
    """
    a an and are as at be by can cannot canceled cancelled command completed could counter crash
    crashed did do does error errors exception exit code exited fail failed failing failure failures
    fatal for from has have in is it job jobs not of on operation or process returned run running
    status step steps test tests the this timeout timed out to unexpected unknown was warning
    warnings were with
    """.split()
)

# Runner/tooling boilerplate printed by many unrelated failures (normalized form).
GENERIC_PHRASES = (
    "process completed with exit code",
    "the operation was canceled",
    "no files were found with the provided path",
    "no artifacts will be uploaded",
)

# Ticket recorded for CI Doctor-proposed entries, which are not tied to a JIRA/GitHub
# ticket (same sentinel as the entries migrated from the CI Doctor knowledge base).
CI_DOCTOR_TICKET = 9999999


def normalize(text: str) -> str:
    """Same normalization as ``workflow_rerun/log_analyzer.py::_clean_up_string``."""
    return re.sub(r"[^A-Za-z0-9]+", " ", text).lower().strip()


def _only_boilerplate(normalized: str) -> bool:
    """True when the string is (part of) generic boilerplate plus only generic words/numbers."""
    if any(normalized in phrase for phrase in GENERIC_PHRASES):
        return True
    remainder = normalized
    for phrase in GENERIC_PHRASES:
        remainder = remainder.replace(phrase, " ")
    if remainder == normalized:
        return False
    return all(word in GENERIC_WORDS or word.isdigit() for word in remainder.split())


def validate_candidate(search_string: str, category: str, signature_hash: str, sample_log_line: str) -> list[str]:
    """Return the list of reasons the candidate is rejected (empty when it is acceptable)."""
    problems: list[str] = []
    if not SIGNATURE_HASH_RE.fullmatch(signature_hash):
        problems.append(f"signature_hash must match {SIGNATURE_HASH_RE.pattern}, got '{signature_hash}'")
    if category not in TRANSIENT_CATEGORIES:
        problems.append(f"category '{category}' is not transient ({', '.join(TRANSIENT_CATEGORIES)})")
    if "|" in search_string or "\n" in search_string or "\r" in search_string:
        problems.append("search string must be a substring of a single log line (no '|' or newlines)")
    if "\n" in sample_log_line or "\r" in sample_log_line:
        problems.append("sample_log_line must be a single log line (no CR or LF)")
    if URL_RE.search(search_string):
        problems.append(
            "search string contains a URL, which pins it to one dependency/version; use the stable "
            "message around it instead (e.g. CMake's 'Each download failed!')"
        )

    normalized = normalize(search_string)
    words = normalized.split()
    if len(words) < MIN_WORDS or len(normalized) < MIN_CHARS:
        problems.append(
            f"normalized string '{normalized}' is too short "
            f"(needs >= {MIN_WORDS} words and >= {MIN_CHARS} characters)"
        )
    if words and all(word in GENERIC_WORDS or word.isdigit() for word in words):
        problems.append(f"normalized string '{normalized}' consists only of generic words")
    elif _only_boilerplate(normalized):
        problems.append(f"normalized string '{normalized}' is runner boilerplate shared by unrelated failures")
    if re.fullmatch(r"[0-9a-f]{8,}", normalized.replace(" ", "")):
        problems.append(f"normalized string '{normalized}' looks like a hash/identifier")

    normalized_line = normalize(sample_log_line)
    if not normalized_line:
        problems.append("sample_log_line is required (verbatim raw log line containing the string)")
    elif normalized and normalized not in normalized_line:
        problems.append("normalized search string is not a substring of the normalized sample_log_line")
    if STEP_HEADER_RE.search(sample_log_line):
        problems.append("sample_log_line is a GitHub Actions step/group header, which appears in every run")
    return problems


def find_overlap(candidate: str, known: Iterable[tuple[str, str]]) -> str | None:
    """Return a description of the first known string overlapping ``candidate``.

    Overlap means either normalized string contains the other: an existing broader
    string already covers the candidate, or the candidate would widen an existing one.
    """
    normalized = normalize(candidate)
    for origin, text in known:
        other = normalize(text)
        if other and (other in normalized or normalized in other):
            return f"overlaps with '{text}' ({origin})"
    return None


def read_errors_file(repo: Any, path: str, ref: str) -> tuple[list[dict[str, Any]], str, str]:
    """Return ``(entries, raw_text, blob_sha)`` of ``errors_to_look_for.json`` at ``ref``."""
    content = repo.get_contents(path, ref=ref)
    text = content.decoded_content.decode("utf-8")
    return json.loads(text), text, content.sha


def as_known(entries: Iterable[dict[str, Any]], origin: str) -> list[tuple[str, str]]:
    return [(origin, entry.get("error_text", "")) for entry in entries if entry.get("error_text")]


def load_open_proposals(repo: Any, base: str, path: str, base_texts: set[str]) -> list[tuple[str, str]]:
    """Search strings added by still-open CI Doctor proposal PRs against ``base``."""
    known = []
    for pull in repo.get_pulls(state="open", base=base):
        if not pull.head.ref.startswith(BRANCH_PREFIX):
            continue
        entries, _, _ = read_errors_file(repo, path, pull.head.sha)
        added = [entry for entry in entries if entry.get("error_text") not in base_texts]
        known.extend(as_known(added, f"open PR #{pull.number}"))
    return known


def previous_proposal(repo: Any, base: str, head_branch: str) -> Any | None:
    """Return any earlier (open, merged or rejected) proposal PR for this signature."""
    owner = repo.full_name.split("/")[0]
    for pull in repo.get_pulls(state="all", base=base, head=f"{owner}:{head_branch}"):
        return pull
    return None


def build_entry(search_string: str, category: str) -> dict[str, Any]:
    """Entry in the ``errors_to_look_for.json`` format."""
    return {"error_text": search_string, "ticket": CI_DOCTOR_TICKET, "category": category}


def render_errors_file(entries: list[dict[str, Any]], original_text: str) -> str:
    """Serialize like the committed file (4-space indent, original trailing whitespace)."""
    trailing = original_text[len(original_text.rstrip()):]
    return json.dumps(entries, indent=4, ensure_ascii=False) + trailing


def build_pr_body(item: dict[str, Any], entry: dict[str, Any], signature_hash: str, path: str, run_url: str) -> str:
    sample = (item.get("sample_log_line") or "").replace("```", "'''")
    title = (item.get("title") or "").strip() or entry["error_text"]
    return (
        "### CI Doctor — proposed workflow-rerunner pattern\n\n"
        "CI Doctor (Merge Queue) classified a failure as **transient** and proposes adding the search "
        f"string below to `{path}`. Once merged, **every failed monitored pipeline whose log contains "
        "this string (after normalization) is re-run automatically**, so please make sure it is specific "
        "to this transient failure.\n\n"
        f"- **Search string**: `{entry['error_text']}`\n"
        f"- **Normalized (what is matched)**: `{normalize(entry['error_text'])}`\n"
        f"- **Category**: {entry['category']}\n"
        f"- **Title**: {title}\n"
        f"- **Signature hash**: `{signature_hash}`\n"
        f"- **Evidence run**: {(item.get('pipeline_url') or '').strip() or 'n/a'}\n"
        f"- **Reason**: {(item.get('reason') or '').strip() or 'n/a'}\n"
        f"- **Proposed by**: {run_url or 'CI Doctor MQ'}\n\n"
        "**Sample log line**\n\n"
        f"```text\n{sample}\n```\n\n"
        "#### Review checklist\n\n"
        "- [ ] The string identifies this transient failure only; it is not a step name, a generic "
        "phrase, or text that also appears in healthy or deterministically failing runs.\n"
        "- [ ] A plain rerun is expected to fix this failure.\n"
        f"- [ ] Optionally replace the placeholder ticket `{CI_DOCTOR_TICKET}` with a real ticket number.\n"
        "- [ ] Edit the string in this PR if it should be narrower; close the PR to reject it "
        "(CI Doctor will not re-propose the same signature).\n"
    )


def write_summary(lines: list[str]) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def skip(message: str) -> None:
    print(message)
    write_summary(["### CI Doctor — rerun pattern proposal skipped", "", message])


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def open_proposal(repo: Any, base: str, head_branch: str, path: str, entry: dict[str, Any],
                  title: str, body: str) -> Any:
    from github import GithubException

    base_sha = repo.get_branch(base).commit.sha
    try:
        repo.create_git_ref(ref=f"refs/heads/{head_branch}", sha=base_sha)
    except GithubException as error:
        if error.status != 422:
            raise
        # Leftover branch without a PR (e.g. an aborted earlier run): reset it to the base.
        repo.get_git_ref(f"heads/{head_branch}").edit(sha=base_sha, force=True)

    entries, text, blob_sha = read_errors_file(repo, path, head_branch)
    content = render_errors_file(entries + [entry], text)
    message = f"[CI] Add rerunner pattern: {title}"[:250]
    repo.update_file(path=path, message=message, content=content, sha=blob_sha, branch=head_branch)
    pr_title = f"{TITLE_PREFIX}Rerun pattern: {title}"[:250]
    return repo.create_pull(title=pr_title, body=body, base=base, head=head_branch, draft=False)


def decorate(pull: Any, labels: list[str], reviewers: list[str]) -> None:
    """Best-effort labels/reviewers: a missing label or reviewer must not lose the PR."""
    if labels:
        try:
            pull.add_to_labels(*labels)
        except Exception as error:
            print(f"Could not add labels {labels}: {error}")
    if reviewers:
        try:
            pull.create_review_request(reviewers=reviewers)
        except Exception as error:
            print(f"Could not request reviewers {reviewers}: {error}")


def main() -> None:
    item = read_agent_item("propose_rerun_pattern")
    search_string = (item.get("rerun_search_string") or "").strip()
    signature_hash = (item.get("signature_hash") or "").strip().lower()
    category = (item.get("category") or "").strip()
    sample_log_line = (item.get("sample_log_line") or "").strip()

    print(f"Rerun pattern candidate: '{search_string}' [{category}] for signature {signature_hash}")
    problems = validate_candidate(search_string, category, signature_hash, sample_log_line)
    if problems:
        skip("Candidate rejected by guardrails:\n" + "\n".join(f"- {p}" for p in problems))
        return

    repository = resolve_repository(os.environ.get("GITHUB_REPOSITORY", ""))
    path = require_env("ERRORS_TO_LOOK_FOR_PATH").strip("/")
    head_branch = f"{BRANCH_PREFIX}{signature_hash}"

    repo = github_client(require_env("GH_TOKEN")).get_repo(repository)
    base = os.environ.get("BASE_BRANCH", "").strip() or repo.default_branch

    base_entries, _, _ = read_errors_file(repo, path, base)
    overlap = find_overlap(search_string, as_known(base_entries, f"{path}@{base}"))
    if overlap:
        skip(f"Candidate already covered by the rerunner list: {overlap}.")
        return

    earlier = previous_proposal(repo, base, head_branch)
    if earlier is not None:
        state = "merged" if earlier.merged else earlier.state
        skip(f"Signature {signature_hash} was already proposed in {earlier.html_url} ({state}); not re-proposing.")
        return

    base_texts = {entry.get("error_text") for entry in base_entries}
    overlap = find_overlap(search_string, load_open_proposals(repo, base, path, base_texts))
    if overlap:
        skip(f"Candidate duplicates a pending rerun pattern proposal: {overlap}.")
        return

    entry = build_entry(search_string, category)
    title = (item.get("title") or "").strip() or search_string
    run_url = ""
    if os.environ.get("GITHUB_RUN_ID"):
        run_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repository}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    body = build_pr_body(item, entry, signature_hash, path, run_url)
    pull = open_proposal(repo, base, head_branch, path, entry, title, body)
    decorate(pull, parse_list(os.environ.get("PR_LABELS", "")), parse_list(os.environ.get("PR_REVIEWERS", "")))

    print(f"Opened rerun pattern proposal {pull.html_url}")
    write_summary(["### CI Doctor — rerun pattern proposal opened", "", f"{pull.html_url}"])


if __name__ == "__main__":
    main()
