---
name: "CI Failure Doctor — Merge Queue"
description: |
  This workflow is an automated CI failure investigator for the GitHub Actions Merge Queue.
  It triggers when monitored merge-queue workflows fail, performs deep analysis of the failure,
  and sends a second escalation alert to Microsoft Teams when the same failure pattern
  has 3 or more occurrences within the last 12 hours.

  The investigation protocol shared with the Post-Commit doctor lives in the
  parameterised `shared/agentic-workflows/ci-doctor-*.md` fragments imported below;
  the body of this file only holds the merge-queue-specific additions.

on:
  workflow_dispatch:
    inputs:
      run_id:
        description: "Workflow run ID to investigate (for manual testing)"
        required: false
      link:
         description: "Link to a workflow to investigate (for manual testing across repositories)"
         required: false
  workflow_run:
    workflows:
      - "Linux (Ubuntu 22.04, Python 3.11)"
      - "Linux (Ubuntu 24.04, Python 3.12)"
      - "Android"
      - "Linux ARM64 (Ubuntu 22.04, Python 3.11)"
      - "Linux (Ubuntu 22.04, ARM64 cross-compilation, Python 3.11)"
      - "Linux Static CC (Ubuntu 22.04, Python 3.11, Clang)"
      - "Linux RISC-V (Ubuntu 22.04, Python 3.10)"
      - "Windows (VS 2022, Python 3.11, Release)"
      - "Windows (VS 2022, Python 3.11, Debug)"
      - "Windows Conditional Compilation (VS 2022, Python 3.11)"
      - "Webassembly"
      - "Manylinux 2_28"
      - "Clang-tidy static analysis (Ubuntu 24.04, Python 3.12, Clang-18, Clang-tidy-18)"
    types:
      - completed
concurrency:
  group: gh-aw-${{ github.workflow }}

# Only trigger for merge-queue failures, or manual workflow_dispatch for testing
if: ${{ github.event_name == 'workflow_dispatch' || (github.event.workflow_run.conclusion == 'failure' && github.event.workflow_run.event == 'merge_group') }}

permissions: read-all

model: claude-sonnet-5
engine:
  id: copilot
network: defaults

imports:
  # Pre-agent steps (log pre-download, PR resolution)
  - shared/agentic-workflows/download-failure-logs.md
  - shared/agentic-workflows/collect-pr-info.md
  # Safe-output jobs
  - shared/agentic-workflows/notify-teams.md
  - shared/agentic-workflows/notify-teams-recurring.md
  - shared/agentic-workflows/remediate-transient-failure.md
  # Shared CI Doctor prompt (inlined before this file's body, in this order).
  # Also wires tools.github, tools.repo-memory (memory/ci-doctor-mq) and the
  # investigations/patterns artifact upload.
  - uses: shared/agentic-workflows/ci-doctor-investigation-protocol.md
    with:
      name: Merge Queue
      kind: merge-queue
      event: merge_group
      slug: mq
  - uses: shared/agentic-workflows/ci-doctor-knowledge-base.md
    with:
      slug: mq
      index_scope_field: pr_number
      pattern_scope_field: affected_prs
      scope_ref: "the PR URL (or null if no PR)"
  - uses: shared/agentic-workflows/ci-doctor-reporting.md
    with:
      name: Merge Queue
      kind: merge-queue
      event: merge_group
      source: merge_queue
      slug: mq

safe-outputs:
  add-comment:
    max: 1              # at most one remediation comment per investigation
    target: "*"         # workflow_run trigger has no PR context; agent supplies the PR number
  report-failure-as-issue:   # defeat the silent "produced no safe outputs" no-op
    - agent_failure
    - missing_safe_outputs
    - timed_out

timeout-minutes: 20

---

# CI Failure Doctor — Merge Queue: Workflow-Specific Instructions

The shared CI Doctor protocol above applies in full. The sections below extend it for the **merge-queue** investigator and take precedence wherever they conflict with it. In addition to the shared mission, this doctor:

- escalates to Microsoft Teams when the same failure pattern recurs **3 or more times within a 12-hour window** (Phase 5.5 / `notify_teams_recurring`),
- posts a remediation comment on the pull request behind the failed merge-queue run (`add_comment`),
- requests automated remediation of **transient** failures (`remediate_transient_failure`),
- maintains the `rerun_search_string` hook consumed by the workflow re-runner.

## Pre-Analysis Data — Pull Request Context

In addition to the pre-downloaded logs:

- **PR info**: `/tmp/gh-aw/agent/ci-doctor/pr-info.json` (structured) and `/tmp/gh-aw/agent/ci-doctor/pr-info.txt` (human-readable) — the pull request behind this merge-queue run, resolved before the session. Includes `pr_number`, `pr_url`, `author`, `title`, `base_branch`, `head_sha`, `labels`, and the list of `changed_files`. **Use these values verbatim for the analysis and safe-outputs.** If the file is empty (`{}`) or `pr-info.txt` says no PR could be associated, treat the PR/Author fields as `not_found` and skip `add_comment`. Prefer the `changed_files` list when scoping PR-diff source inspection (Phase 4).

This file is the **only** source of PR context for this doctor (Phase 3 step 4, Phase 4 PR-scoped diffs, and the `pr_number` / `pr_url` / `author` fields of `notify_teams`). Do not re-derive these values from the run metadata yourself.

## Phase 5 — Merge-Queue Additions to the Pattern Record

The merge-queue pattern schema (`${GITHUB_WORKSPACE}/.github/ci-doctor-mq/schemas/pattern.schema.json`) has one extra **required** field, `rerun_search_string`, on top of the shared record computed in Phase 5 step 2, Step C. Apply the following on top of the shared procedure.

### Step C (addendum) — set `rerun_search_string`

In both branches of the Step C pseudocode additionally set:

~~~pseudocode
record.rerun_search_string = RERUN_SEARCH_STRING   ← compute per Step C.1 below (recompute on every update)
~~~

### Step C.1 — Compute and verify `RERUN_SEARCH_STRING` (the workflow_rerunner hook)

This field feeds the automated rerunner (`.github/scripts/workflow_rerun`), which greps the failed run's logs for this string and, on a match, re-runs the failed jobs — exactly like the static entries in `.github/scripts/workflow_rerun/errors_to_look_for.json`. A rerun only helps for **transient** failures, and a string that never matches is useless:

- IF `category` is NOT one of `Flaky Test`, `Infrastructure`, `Network`, or `External Service` (i.e. a deterministic `Code Issue`/`Dependencies`/`Configuration` failure a restart cannot fix): set `RERUN_SEARCH_STRING = null`. A search string here would make the rerunner loop on an unfixable failure.
- ELSE: set `RERUN_SEARCH_STRING` to a **short, stable substring taken verbatim from a single raw log line** that uniquely identifies this failure (e.g., `Could not resolve host`, `Connection reset by peer`, `runner has received a shutdown signal`). Strip everything volatile that changes between runs or across jobs: absolute paths, line/column numbers, hex addresses, PIDs, timestamps, run IDs, commit SHAs, tmp dirs, UUIDs, and embedded job/runner/OS/shard names.
- Always recompute on update so a category change (e.g. `Flaky Test` → `Code Issue`) correctly clears or sets the string.

**Verify a non-null value against the rerunner's exact matcher** ([`log_analyzer.py`](../scripts/workflow_rerun/log_analyzer.py)): it normalizes a string by replacing each run of non-alphanumeric chars (`[^A-Za-z0-9]+`) with one space, lower-casing, and stripping (so `Could not resolve host: github.com` → `could not resolve host github com`), then checks whether the normalized string is a plain substring of a **single** normalized log line (matches never span line breaks). Confirm the normalized `RERUN_SEARCH_STRING` is a substring of a normalized real log line from the pre-downloaded logs, is non-empty after normalization, and is specific enough to avoid matching benign lines (no bare `error`/`failed`/`warning`). If you cannot satisfy this, set `RERUN_SEARCH_STRING = null` rather than store an unverified guess.

### Step E (addendum) — extra read-back checks

- `rerun_search_string` is present and is either a non-empty string (only for the transient categories `Flaky Test`/`Infrastructure`/`Network`/`External Service`) or `null` (for every other category)
- if `rerun_search_string` is non-null, it still satisfies the Step C.1 verification (normalized substring of a single actual failure-log line, non-empty after normalization); otherwise set it to `null`

### Step 5 — Backfill Missing `rerun_search_string` Fields (MANDATORY sweep, every run)

Some pattern files were written before `rerun_search_string` existed and are missing the field entirely (as opposed to having it explicitly set to `null`). Every time you run Phase 5, after step 2 has written/updated the current failure's pattern file, sweep **all** other files in `/tmp/gh-aw/repo-memory/default/mq/patterns/` and backfill any that lack the key:

- For each `<signature-hash>.json` file in the patterns directory (including ones untouched by this run):
  1. Read and parse the file.
  2. If the `rerun_search_string` key is **already present** (even if its value is `null`), skip it — do not touch a file that already conforms to the schema.
  3. If the key is **missing**, derive it from the file's own `signature` field (`<normalized-error>|<category>`) using the same rules as Step C.1, without needing the original failure logs:
     - Split `signature` on the last `|` to recover `<normalized-error>` and `<category>`.
     - IF `<category>` is NOT one of `Flaky Test`, `Infrastructure`, `Network`, or `External Service`: set `rerun_search_string = null`.
     - ELSE: set `rerun_search_string = <normalized-error>` verbatim (the signature's error component was already stripped of volatile tokens per Step A, so no further normalization is needed). Since the original raw logs are unlikely to still be available for a backfill, you do NOT need to re-verify it against a live log line as Step C.1 otherwise requires — only confirm it is non-empty and not just generic noise (e.g., not solely `error`, `failed`, or `warning`). If it fails that minimal check, set it to `null` instead of guessing.
  4. Write the updated object back to the same file, changing only the `rerun_search_string` key — do not alter `count`, timestamps, `signature`, `signature_hash`, or any other field.
  5. Validate the rewritten file against `pattern.schema.json` (see the validation procedure in the "Artefact schemas" block) and confirm `rerun_search_string` is now present as either a non-empty string or `null`.

This step must never modify a file's `count`, `first_seen`, `last_seen`, or timestamp arrays — its only job is adding the missing key.

## Phase 5.5 — Recurring Failure Escalation Check

After updating the pattern database (Phase 5 step 2), check whether the current failure's pattern has occurred **3 or more times in the last 12 hours**.

**Recurrence detection procedure (follow EXACTLY):**

~~~pseudocode
1. FILE_PATH = /tmp/gh-aw/repo-memory/default/mq/patterns/<signature-hash>.json
2. Read FILE_PATH → parse as JSON into `pattern`
3. NOW = current UTC time
4. CUTOFF = NOW - 12 hours
5. recent_hits = [ts for ts in pattern.recent_timestamps if ts >= CUTOFF]
6. recent_count = len(recent_hits)

IF recent_count >= 3:
    → collect affected_prs (up to 10)
    → collect recent_run_urls (up to 10)
    → format both as markdown bullet lists
    → call notify_teams_recurring with:
        title = same as notify_teams.title
        failed_workflow = same as notify_teams.failed_workflow
        pipeline_url = URL of the current failed run
        recent_count = str(recent_count)  (e.g., "3", "4", "5")
        description = concise gist of the recurring problem
        affected_prs = markdown list
        recent_run_urls = markdown list
ELSE:
    → do NOT call notify_teams_recurring
~~~

**Important:** If `pattern.recent_timestamps` is empty or missing, it means the pattern file was written incorrectly in Phase 5 step 2. Go back and fix the write — the current timestamp MUST appear in `recent_timestamps`. Do NOT skip the escalation check just because the array is empty.

## Phase 6 — Additional Deliverables

- When the failure is associated with a PR in the merge queue, post a remediation comment on that PR with the failed pipeline name/link, a short failure description, and a short possible remedy (see `add_comment` field guidance below)
- When the investigation concludes the failure is **transient** (Infrastructure / Flaky Test / Network / External Service), request automated remediation by calling `remediate_transient_failure` (see its decision guidance below).

## Phase 7 — Additional String-Encoding Checklist

For `notify_teams_recurring`:
- `title`, `failed_workflow`, `pipeline_url`, `description`, `affected_prs`, `recent_run_urls` — non-empty strings
- `recent_count` — string-encoded positive integer, e.g. `"3"` (NOT `3`)

## Output Requirements — Merge-Queue Additions

### `notify_teams` — merge-queue specifics

- **`pr_number`** / **`pr_url`** — Read both directly from the pre-collected `/tmp/gh-aw/agent/ci-doctor/pr-info.json` (`pr_number`, `pr_url`). Provide them together whenever that file identifies a PR. Omit both only if the file is empty (`{}`) or no PR could be resolved.
- **`author`** — Read from the `author` field of `/tmp/gh-aw/agent/ci-doctor/pr-info.json`. Omit only if the file has no PR / empty `author`.
- **`description`** — Insert the following section into the shared template **right after `### Failure Details`**, stating whether automated remediation of a transient failure was requested:

```markdown
### Automatic Remediation

- If you called `remediate_transient_failure`: `✅ Remediation requested` followed by the one-line `reason` you passed. Note that the job itself decides at run time whether to re-run the failed jobs (PR still in the queue) or re-add the PR to the merge queue (PR dropped) — you do not choose or report which.
- Otherwise: `❌ Not triggered` followed by a short justification (e.g. deterministic code failure that neither a restart nor a re-queue can fix).
```

Additionally, if Phase 5.5 determines the same failure has occurred 3 or more times in the last 12 hours, call the `notify_teams_recurring` safe-output tool exactly once with the escalation details.

Additionally, **when the failure is associated with a PR in the merge queue**, post a remediation comment on that PR by calling the `add_comment` safe-output tool exactly once (see field guidance below). If no PR can be identified, skip the comment.

### `add_comment` field guidance

Post a concise, actionable remediation comment on the affected merge-queue PR so the author has the context and next steps. Call `add_comment` **at most once per investigation** and **only** when a PR can be identified.

- **`item_number`** (required) — The number of the affected PR in the merge queue (the `pr_number` from `/tmp/gh-aw/agent/ci-doctor/pr-info.json`, i.e. the same value reported as `notify_teams.pr_number`). This is required because the `workflow_run` trigger carries no PR context; the comment cannot be posted without it. If `pr-info.json` identifies no PR, skip the comment.

- **`body`** (required) — Markdown comment body. Keep it focused and short. GitHub renders standard Markdown here (headings, bold, inline code, fenced code blocks with backticks, lists, links). Use this structure:

```markdown
### CI Doctor — Merge Queue failure on this PR

**Pipeline**: [<failed_workflow name>](<pipeline_url>)
**Failure**: <one-line summary, same as notify_teams.title>
**Automatic remediation**: <one of: `✅ Requested (reason: <reason>)` when you called `remediate_transient_failure` — the job re-runs the failed jobs if the PR is still queued, or re-adds the PR to the merge queue if it was dropped; `❌ Not triggered — <short reason, e.g. deterministic code failure>` otherwise>

#### Possible remedy

<1–4 concrete, actionable steps to fix or work around the failure, based on the
root-cause analysis. Reference specific files/lines from the logs when available.>

#### What happened

<1–2 sentence plain-language description of the failure: which job(s) failed and the key error.>

<If repo-memory shows this is a known/recurring pattern, add one line noting how
many times it has been seen and link the most recent prior failure run.>
```

Source the comment content directly from the investigation you already produced:
  * **Pipeline name + link** come from `failed_workflow` and `pipeline_url`.
  * **Failure summary** matches `notify_teams.title`.
  * **What happened** is a condensed version of the Root Cause Analysis (Phase 4 / Phase 6).
  * **Possible remedy** comes from your Recommended Actions, refined with any matching pattern data from repo-memory (`/tmp/gh-aw/repo-memory/default/mq/patterns/` and `/tmp/gh-aw/repo-memory/default/mq/investigations/`). If a prior pattern exists, prefer the remedy that resolved it before.

Do not duplicate the full Teams description in the comment — keep it to the pipeline reference, a short possible remedy, and a failure description.

### `notify_teams_recurring` field guidance

This notification is **only** sent when the same failure has occurred 3 or more times in the last 12 hours (Phase 5.5). It provides a condensed escalation alert separate from the detailed per-failure investigation.

- **`title`** — Same short description as `notify_teams.title`.
- **`failed_workflow`** — Same as `notify_teams.failed_workflow`.
- **`pipeline_url`** — URL of the current (latest) failed run.
- **`recent_count`** — Number of occurrences of this failure in the last 12 hours, including the current run (e.g., `"3"`, `"5"`).
- **`description`** — Concise gist (3–5 sentences) of the recurring problem: what keeps failing, suspected root cause, and a recommended escalation action.
- **`affected_prs`** — Markdown bullet list of PRs affected by this failure in the last 12 hours (up to 10, e.g., `- [#1234](url)`). If no PRs can be identified, write "No PR information available."
- **`recent_run_urls`** — Markdown bullet list of failure run URLs from the last 12 hours (up to 10, e.g., `- [Run 56789](url)`).

### `remediate_transient_failure` decision guidance

Call the `remediate_transient_failure` safe-output tool **only** when your Root Cause Analysis concludes the failure is transient and a plain retry is likely to clear it — the `Infrastructure`, `Flaky Test`, `Network`, or `External Service` categories.

**Do NOT** call it for deterministic failures a retry cannot fix — `Code Issue`, `Dependencies`, or `Configuration` categories (compilation errors, assertion failures, missing symbols, bad workflow config). When in doubt, do not call it.

When you call `remediate_transient_failure`, the job resolves the PR behind the analysed run, reads its **live** merge-queue status at action time, and picks the remedy itself:
- **PR still in the queue** (or no PR is associated) → it re-runs only the failed jobs.
- **PR dropped from the queue** → it re-adds the PR to the merge queue (idempotent: it skips PRs that are already merged, closed, draft, or previously re-added).
- **PR already merged, or status indeterminate** → it takes no action (fail safe).

Call it **at most once** per investigation. It re-runs only the **failed** jobs (passing jobs are untouched) and refuses to re-run a run already on its second attempt, to avoid restart loops.

Provide:

- **`run_id`** (required) — Numeric ID of the analysed run: `${{ github.event.workflow_run.id }}` for merge-queue triggers, or the `run_id` input for `workflow_dispatch`. Pass as a numeric string.
- **`repository`** (optional) — `owner/repo` of the analysed run. Omit to default to the current repository.
- **`reason`** (required) — One-line justification for why the failure is transient, matching the cause identified in the investigation.

This tool is independent of the notifications: still call `notify_teams` (and `add_comment` / `notify_teams_recurring` when applicable) as usual. A remediation request does not replace the investigation report.

Whenever you decide about remediation (whether or not you trigger it), you MUST record the outcome in both the Teams message (the `### Automatic Remediation` section of `notify_teams.description`) and, when a PR comment is posted, the `**Automatic remediation**` line of the `add_comment` body. Keep both consistent with the actual `remediate_transient_failure` call.

## Mandatory Output Requirement — Merge-Queue Safe Outputs

In addition to `notify_teams`, `noop` and `missing_data` from the shared protocol, this workflow may call (all numeric-looking fields, including `recent_count`, as JSON strings):

- **`notify_teams_recurring`**: Send a recurring-failure escalation alert. Call this **only** if Phase 5.5 determines that there are 3+ occurrences in the last 12 hours. Call at most once per run.
- **`add_comment`**: Post a remediation comment on the affected merge-queue PR. Call this **only** when the failure is associated with a PR (provide `item_number` and `body`). Call at most once per run.
- **`remediate_transient_failure`**: Remediate a transient merge-queue failure. Call this **only** when the failure is transient (Infrastructure / Flaky Test / Network / External Service). The job resolves the PR's live merge-queue status and decides whether to re-run the failed jobs or re-add the PR to the queue — you do not choose. Call at most once per run.

**Valid call combinations:**
- `notify_teams` alone — standard investigation with no identifiable PR, fewer than 3 occurrences in the last 12 hours.
- `notify_teams` + `add_comment` — standard investigation where the failure is tied to a PR in the merge queue.
- `notify_teams` + `notify_teams_recurring` (+ `add_comment` when a PR is identified) — standard investigation AND 3+ occurrences in the last 12 hours.
- Any of the `notify_teams` combinations above **+ `remediate_transient_failure`** — a transient failure; the job decides at run time whether to re-run the failed jobs or re-add the PR to the merge queue.
- `noop` alone — no investigation needed.
- `missing_data` alone — investigation blocked by missing data.
