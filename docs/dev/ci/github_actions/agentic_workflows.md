# Agentic Workflows

OpenVINO CI includes a set of **agentic workflows** — GitHub Actions workflows that hand control to
an AI agent to perform an investigative or maintenance task, rather than running a fixed sequence of
shell commands. At the moment there are two of them, both dedicated to diagnosing CI failures:

* [`ci-doctor.md`](../../../../.github/workflows/ci-doctor.md) — **CI Doctor**, an on-demand
  investigator for a pull request.
* [`ci-doctor-mq.md`](../../../../.github/workflows/ci-doctor-mq.md) — **CI Doctor (Merge Queue)**,
  an automatic investigator for merge-queue failures.

This document explains what they are, how they are built and invoked, the algorithm each one follows,
and the reusable jobs from [`.github/workflows/shared`](../../../../.github/workflows/shared/agentic-workflows)
that they share.

## Table of Contents

* [What is an agentic workflow](#what-is-an-agentic-workflow)
  * [The gh-aw framework](#the-gh-aw-framework)
  * [Source files and compiled lock files](#source-files-and-compiled-lock-files)
* [Workflows at a glance](#workflows-at-a-glance)
* [CI Doctor (pull request, on-demand)](#ci-doctor-pull-request-on-demand)
* [CI Doctor — Merge Queue (automatic)](#ci-doctor--merge-queue-automatic)
* [Shared reusable jobs](#shared-reusable-jobs)
* [Setup and infrastructure](#setup-and-infrastructure)
  * [Log pre-download and caching](#log-pre-download-and-caching)
  * [Repo-memory and the pattern database](#repo-memory-and-the-pattern-database)
  * [Pattern matching and recurrence detection](#pattern-matching-and-recurrence-detection)
  * [Automated re-run integration](#automated-re-run-integration)
  * [Secrets and permissions](#secrets-and-permissions)
* [Maintaining the workflows](#maintaining-the-workflows)
* [See also](#see-also)

## What is an agentic workflow

A traditional workflow describes *exactly* what to run: check out the code, install dependencies, run
a script. An **agentic workflow** instead describes a *goal* and a *protocol* in natural language, and
delegates the execution to an AI agent. The agent decides which tools to call —
reading logs, searching the repository, querying the GitHub API — to accomplish the task, subject to
the guardrails written into the workflow.

In OpenVINO both agentic workflows are **CI failure doctors**: they read the logs of a failed pipeline,
localise the root cause, classify the failure, and produce an actionable report (a PR comment or a
Microsoft Teams notification).

### The gh-aw framework

The workflows are authored with [GitHub Agentic Workflows (`gh-aw`)](https://github.github.com/gh-aw/introduction/overview/).
A `gh-aw` workflow is a Markdown file with a YAML frontmatter block and a natural-language body:

* **Frontmatter** configures the workflow the same way a normal GitHub Actions workflow is configured —
  `on:` triggers, `permissions:`, `concurrency:`, plus `gh-aw`-specific keys:
  * `engine:` — the agent runtime (`copilot`) and `model:` (`claude-sonnet-4.6`).
  * `network:` — the network policy enforced by a firewall proxy around the agent.
  * `tools:` — which tool servers the agent may use (for example the GitHub MCP server toolsets, and
    the `repo-memory` store).
  * `safe-outputs:` — the only side effects the agent is allowed to produce. The agent cannot post a
    comment or send a notification directly; it emits a structured *safe output* that a trusted,
    non-agent job then acts on.
  * `imports:` — shared job/step fragments pulled in from
    [`.github/workflows/shared`](../../../../.github/workflows/shared/agentic-workflows).
* **Body** is the agent's prompt: the mission, the investigation protocol, the output format, and the
  guardrails (for example, "read at most 10 source files" or "never write to the knowledge base").

### Source files and compiled lock files

`gh-aw` does not run the `.md` file directly. It **compiles** each `.md` into a standard GitHub Actions
workflow with the `.lock.yml` extension:

| Source (edit this) | Compiled (generated) |
| --- | --- |
| [`ci-doctor.md`](../../../../.github/workflows/ci-doctor.md) | [`ci-doctor.lock.yml`](../../../../.github/workflows/ci-doctor.lock.yml) |
| [`ci-doctor-mq.md`](../../../../.github/workflows/ci-doctor-mq.md) | [`ci-doctor-mq.lock.yml`](../../../../.github/workflows/ci-doctor-mq.lock.yml) |

The `.lock.yml` file is what GitHub Actions actually executes. It is **auto-generated — do not edit it
by hand**. After changing a `.md` source (or any imported shared file), regenerate the lock file:

```bash
gh aw compile
```

See [Maintaining the workflows](#maintaining-the-workflows) for details.

## Workflows at a glance

| | **CI Doctor** | **CI Doctor — Merge Queue** |
| --- | --- | --- |
| Source | [`ci-doctor.md`](../../../../.github/workflows/ci-doctor.md) | [`ci-doctor-mq.md`](../../../../.github/workflows/ci-doctor-mq.md) |
| Trigger | On demand: `/ci-doctor` comment on a PR | Automatic: a monitored workflow finishes a `merge_group` run with `failure` |
| Scope | **Every** failed pipeline on the PR head commit | A **single** failed merge-queue run |
| Knowledge base | Reads the MQ pattern database (read-only) | **Writes** the pattern database (investigations + patterns) |
| Output | One consolidated PR comment | Microsoft Teams notification (+ optional PR comment, re-run, re-queue) |
| Remediation | Report only | Can re-run failed jobs, re-add the PR to the merge queue, escalate recurring failures |
| Engine / model | `copilot` / `claude-sonnet-4.6` | `copilot` / `claude-sonnet-4.6` |

## CI Doctor (pull request, on-demand)

**What it is** An on-demand assistant that investigates *all* currently failing CI pipelines on a
pull request and summarises them in a single comment.

**How it is invoked** A user posts the `/ci-doctor` slash command as a comment on the pull request.
Invocation is restricted by an `if:` guard in the frontmatter to an explicit allow-list of maintainers,
so arbitrary users cannot trigger agent runs:

```yaml
on:
  slash_command:
    name: ci-doctor
    events: [pull_request_comment]

if: ${{ contains(fromJSON('["akashchi","mryzhov","akladiev","ababushk","dorloff","as-suvorov"]'), github.actor) }}
```

**What it does / benefits**

* Aggregates *every* failed pipeline on the PR head commit into **one** comment, so the author does not
  have to open each failed run individually.
* For each failure it reports a category, the failed job(s), key log excerpts, a root-cause explanation,
  and a concrete remedy.
* Cross-references the merge-queue knowledge base (read-only) to flag failures that are already known
  recurring issues, and prefers a remedy that resolved them before.
* Bounded and safe: it only ever produces a single PR comment (`safe-outputs.add-comment: max: 1`),
  and its source-code inspection is capped (≤ 10 files, ≤ 5 searches, PR-diff first).

**Algorithm.** The body defines a five-phase protocol:

1. **Initial triage** — read the pre-downloaded summary of failed pipelines; if there are none (or the
   command was not run on a PR), stop with a `noop`.
2. **Deep log analysis** — for each failed pipeline, start from the pre-located error *hints*, then read
   ~50 lines of context around each hint before loading a whole log.
3. **Historical context** — match each failure against the merge-queue pattern database to detect known
   recurring issues (strictly read-only).
4. **Root-cause investigation** — classify the failure into one of seven categories and localise it,
   within the source-inspection safeguards.
5. **Reporting** — post one consolidated comment (via the `add_comment` safe output) with an overview
   table and a per-pipeline section; otherwise `noop`/`missing_data`.

## CI Doctor — Merge Queue (automatic)

**What it is** An automatic, always-on investigator that reacts to merge-queue CI failures, builds a
persistent knowledge base of failure patterns, and escalates when the same failure keeps recurring.

**How it is triggered** It runs on the `workflow_run: completed` event of a fixed list of monitored
workflows (the Linux/Windows/Android/ARM/RISC-V/… validation pipelines), and only proceeds when the run
was a **merge-queue** run that **failed**:

```yaml
on:
  workflow_run:
    workflows: ["Linux (Ubuntu 22.04, Python 3.11)", "Windows (VS 2022, Python 3.11, Release)", ...]
    types: [completed]

if: ${{ github.event_name == 'workflow_dispatch' || (github.event.workflow_run.conclusion == 'failure' && github.event.workflow_run.event == 'merge_group') }}
```

A `workflow_dispatch` entry (with `run_id` or `link` inputs) is also provided for manual testing.

**What it does / benefits**

* **Root-cause analysis** of a single failed merge-queue run: category, failed jobs, key errors,
  explanation, confidence.
* **Knowledge building**: every investigation is written to a persistent, cross-run knowledge base so
  recurring failures accumulate statistics (`count`, `first_seen`, `last_seen`) over time.
* **Escalation**: when the same failure signature is seen **3 or more times within 12 hours**, it sends
  a second, higher-visibility Teams alert.
* **Automated remediation**: when a failure is transient it can either re-run only the failed jobs, or —
  if GitHub already dropped the PR from the queue — re-add the PR to the merge queue. These two actions
  are mutually exclusive.

**Algorithm** The body defines a seven-phase protocol:

1. **Trigger detection & triage** — verify the run is a failed `merge_group` run; otherwise exit.
2. **Deep log analysis** — hint-first log reading, same as CI Doctor.
3. **Historical context** — search prior investigations for similar signatures.
4. **Root-cause investigation** — classify into one of the seven categories, within the source-inspection
   safeguards.
5. **Pattern storage** — write the investigation record, update the append-only investigations index, and
   read-modify-write the per-signature pattern file (all validated against JSON schemas). See
   [Repo-memory and the pattern database](#repo-memory-and-the-pattern-database).
   * **5.5 Recurrence check** — if ≥ 3 hits in the last 12 h, prepare a recurring-failure escalation.
6. **Reporting** — compose the investigation report.
7. **Output validation** — normalise the safe-output payload (all numeric-looking fields must be
   strings) before emitting it.

The run always ends by calling exactly one (or a valid combination) of the safe outputs: `notify_teams`,
`notify_teams_recurring`, `add_comment`, `rerun_failed_jobs`, `readd_to_merge_queue`, `noop`, or
`missing_data`.

## Shared reusable jobs

Common steps and safe-output jobs are factored into
[`.github/workflows/shared/agentic-workflows`](../../../../.github/workflows/shared/agentic-workflows)
and pulled into a workflow via the `imports:` key. Imported `steps:` are prepended to the importing
workflow, and imported `safe-outputs.jobs:` become callable safe outputs for the agent.

| Shared file | Kind | Used by | Purpose |
| --- | --- | --- | --- |
| [`download-failure-logs.md`](../../../../.github/workflows/shared/agentic-workflows/download-failure-logs.md) | Pre-agent step | both | Pre-download failed logs and pre-locate error hints before the agent starts. |
| [`notify-teams.md`](../../../../.github/workflows/shared/agentic-workflows/notify-teams.md) | Safe-output job | MQ | Send the investigation summary to Microsoft Teams; upload the statistics artifact. |
| [`notify-teams-recurring.md`](../../../../.github/workflows/shared/agentic-workflows/notify-teams-recurring.md) | Safe-output job | MQ | Send a recurring-failure escalation alert to Teams. |
| [`rerun-failed-jobs.md`](../../../../.github/workflows/shared/agentic-workflows/rerun-failed-jobs.md) | Safe-output job | MQ | Re-run only the failed jobs of the analysed run (transient failures). |
| [`readd-to-merge-queue.md`](../../../../.github/workflows/shared/agentic-workflows/readd-to-merge-queue.md) | Safe-output job | MQ | Re-add a dropped PR to the merge queue (transient failures). |

**`download-failure-logs.md`** is a *step* fragment (it has no `on:` trigger). It auto-detects its mode
from the environment: **run mode** (`RUN_ID` set) analyses a single run; **PR mode** (`PR_NUMBER` set)
analyses every failed run on a PR head commit. In both modes it writes job logs, per-job error *hint*
files, and a `summary.txt` under `/tmp/gh-aw/agent/ci-doctor/` so the agent can start from a compact
summary instead of downloading logs itself.

**`notify-teams.md`** defines the `notify-teams` safe-output job. It reads the agent's `notify_teams`
item, renders an Adaptive Card (title, facts, description, and a pattern-database statistics table),
POSTs it to the `TEAMS_WEBHOOK_URL`, and uploads the full statistics JSON/Markdown as the
`ci-doctor-mq-statistics` artifact.

**`notify-teams-recurring.md`** defines the `notify-teams-recurring` job, used only when a failure has
recurred ≥ 3 times in 12 hours. It renders a condensed escalation card listing the affected PRs and the
recent failure runs.

**`rerun-failed-jobs.md`** defines the `rerun-failed-jobs` job (`permissions: actions: write`). It calls
the GitHub `rerun-failed-jobs` API for the analysed run, with a loop guard that refuses to re-run a run
that already has more than one attempt.

**`readd-to-merge-queue.md`** defines the `readd-to-merge-queue` job. It re-adds a dropped PR to the
queue via `gh pr merge` using the `MERGE_QUEUE_TOKEN` secret (the default `GITHUB_TOKEN` cannot
re-trigger `merge_group` check runs). It is idempotent and loop-safe: it skips PRs that are merged,
closed, draft, or already carry the CI Doctor re-add marker comment.

## Setup and infrastructure

### Log pre-download and caching

Downloading and scanning large CI logs from inside the agent is slow and burns the agent's context
window. Instead, the imported [`download-failure-logs.md`](../../../../.github/workflows/shared/agentic-workflows/download-failure-logs.md)
step runs **before** the agent and produces a compact, pre-digested view under
`/tmp/gh-aw/agent/ci-doctor/`:

* `logs/job-<id>.log` — the full log of each failed job.
* `filtered/*-hints.txt` — pre-located error lines (matched with a generic error/FAIL/panic/fatal regex),
  so the agent can jump straight to the interesting line numbers.
* `summary.txt` — the entry point the agent reads first: the failed jobs/pipelines, file locations, and
  the first few hint matches.

The agent is instructed to read the summary and hint files first, then read ~50 lines of context around
each hint, and only load a full log if the hints are insufficient.

### Repo-memory and the pattern database

CI Doctor — Merge Queue persists its knowledge across runs using the `repo-memory` tool, which mounts a
dedicated Git branch (`memory/ci-doctor-mq`) at `/tmp/gh-aw/repo-memory/default/`. Anything written
there survives indefinitely; anything written elsewhere is discarded when the runner is torn down. The
workflow uses the `mq/` subdirectory to isolate merge-queue data:

* `mq/investigations/<timestamp>-<run-id>.json` — one record per investigation.
* `mq/investigations/index.json` — an append-only aggregate index of every investigation.
* `mq/patterns/<signature-hash>.json` — one record per failure signature, with `count`, `first_seen`,
  `last_seen`, `recent_timestamps`, and the affected runs/PRs.

Every artifact conforms to a committed JSON Schema and is validated immediately after being written:

| Artifact | Schema |
| --- | --- |
| Investigation record | [`investigation.schema.json`](../../../../.github/ci-doctor-mq/schemas/investigation.schema.json) |
| Pattern record | [`pattern.schema.json`](../../../../.github/ci-doctor-mq/schemas/pattern.schema.json) |
| Investigations index | [`index.schema.json`](../../../../.github/ci-doctor-mq/schemas/index.schema.json) |

CI Doctor (the PR workflow) mounts the same branch **read-only** to detect known recurring issues, and
never writes to it.

### Pattern matching and recurrence detection

Failures are de-duplicated by a **signature hash** so that the *same* underlying error collapses into a
single pattern regardless of how many jobs hit it. The hash is derived from:

1. the **normalised primary error message** — with volatile tokens stripped out (absolute paths,
   line/column numbers, hex addresses, PIDs, timestamps, run IDs, commit SHAs, temp dirs, UUIDs, and any
   embedded job / runner / OS / shard names), and
2. the **failure category** (one of seven fixed values).

The failed job name is deliberately **excluded** from the hash — keying on it would split one error into
a separate pattern per job and break recurrence counting.

Each time a signature is seen, its pattern file is read-modified-written: `count` is incremented,
`last_seen` and `recent_timestamps` are updated, and `first_seen` is preserved. Phase 5.5 then counts
how many `recent_timestamps` fall within the last 12 hours; **3 or more** triggers the recurring-failure
Teams escalation.

### Automated re-run integration

Pattern records also carry a `rerun_search_string` that feeds the static
[`workflow_rerun`](../../../../.github/scripts/workflow_rerun) tooling. For **transient** categories
(`Flaky Test`, `Infrastructure`, `Network`, `External Service`) the agent stores a short, stable
substring taken verbatim from a real failure log line; for deterministic categories it stores `null` so
the re-runner never loops on an unfixable failure. The string is verified against the exact matcher in
[`log_analyzer.py`](../../../../.github/scripts/workflow_rerun/log_analyzer.py) — the same mechanism that
backs the static entries in
[`errors_to_look_for.json`](../../../../.github/scripts/workflow_rerun/errors_to_look_for.json).

### Secrets and permissions

Both workflows run with `permissions: read-all` for the agent itself; each *safe-output* job requests
only the narrow permission it needs. The workflows rely on the following secrets:

| Secret | Used by | Purpose |
| --- | --- | --- |
| `TEAMS_WEBHOOK_URL` | `notify-teams`, `notify-teams-recurring` | Microsoft Teams incoming webhook. |
| `MERGE_QUEUE_TOKEN` | `readd-to-merge-queue` | PAT / App token with `contents: write` + `pull_requests: write` to re-queue a PR (the default token cannot re-trigger `merge_group` runs). |
| `GITHUB_TOKEN` | log download, `rerun-failed-jobs` | Standard GitHub API access. |

## Maintaining the workflows

1. Edit the **`.md` source** (or an imported file under
   [`.github/workflows/shared`](../../../../.github/workflows/shared/agentic-workflows)) — never the
   generated `.lock.yml`.
2. Recompile:

   ```bash
   gh aw compile
   ```

   This regenerates `ci-doctor.lock.yml` / `ci-doctor-mq.lock.yml`. The lock file carries a hash of the
   frontmatter and body, so unrelated edits will not always change it.
3. Commit **both** the `.md` and the regenerated `.lock.yml` together.

When changing the merge-queue knowledge-base format, update the matching schema under
[`.github/ci-doctor-mq/schemas`](../../../../.github/ci-doctor-mq/schemas) so the in-workflow validation
stays in sync.

The repository ships an `ov-agentic-workflows` [agent skill](../../../../.agents/skills/ov-agentic-workflows/SKILL.md)
that captures these editing rules and common tasks. When you work on these workflows with an AI coding
assistant (such as GitHub Copilot), it loads the skill automatically to apply the correct procedure and
guardrails.

## See also

* [Overview of the OpenVINO GitHub Actions CI](./overview.md)
* [Reusable Workflows](./reusable_workflows.md)
* [GitHub Actions security guidelines](./security.md)
* [GitHub Agentic Workflows documentation](https://github.github.com/gh-aw/introduction/overview/)
