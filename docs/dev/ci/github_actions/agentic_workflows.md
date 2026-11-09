# Agentic Workflows

OpenVINO CI includes four **agentic workflows** — GitHub Actions workflows that hand control to
an AI agent to perform an investigative or maintenance task, rather than running a fixed sequence of
shell commands. The four workflows listed below include automatic investigation and maintenance:

* [`ci-doctor.md`](../../../../.github/workflows/ci-doctor.md) — **CI Doctor**, an on-demand
  investigator for a pull request.
* [`ci-doctor-mq.md`](../../../../.github/workflows/ci-doctor-mq.md) — **CI Doctor (Merge Queue)**,
  an automatic investigator for merge-queue failures.
* [`ci-doctor-post-commit.md`](../../../../.github/workflows/ci-doctor-post-commit.md) — **CI Doctor
  (Post-Commit)**, an automatic investigator for post-commit (push) failures that only collects and
  reports (it never re-runs or re-queues pipelines).
* [`ci-doctor-remediation.md`](../../../../.github/workflows/ci-doctor-remediation.md) — **CI Doctor
  (Weekly Remediation)**, a scheduled maintenance job that consumes the two doctors' knowledge base
  (read-only) and opens up to 3 draft remediation pull requests plus a Markdown report artifact.

The Merge Queue doctor is paired with the deterministic
[`ci-doctor-mq-readd.yml`](../../../../.github/workflows/ci-doctor-mq-readd.yml) workflow. It does no
analysis: it checks for a transient-eligibility marker when a PR is dequeued, and retries after the MQ
doctor completes to cover the marker-timing race.

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
* [CI Doctor MQ — Re-add (deterministic)](#ci-doctor-mq--re-add-deterministic)
* [CI Doctor — Post-Commit (automatic)](#ci-doctor--post-commit-automatic)
* [Shared reusable jobs and prompt fragments](#shared-reusable-jobs-and-prompt-fragments)
  * [Shared jobs and steps](#shared-jobs-and-steps)
  * [Shared prompt fragments](#shared-prompt-fragments)
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

In OpenVINO all agentic workflows are **CI failure doctors**: they read the logs of a failed pipeline,
localise the root cause, classify the failure, and produce an actionable report (a PR comment or a
Microsoft Teams notification).

### The gh-aw framework

The workflows are authored with [GitHub Agentic Workflows (`gh-aw`)](https://github.github.com/gh-aw/introduction/overview/).
A `gh-aw` workflow is a Markdown file with a YAML frontmatter block and a natural-language body:

* **Frontmatter** configures the workflow the same way a normal GitHub Actions workflow is configured —
  `on:` triggers, `permissions:`, `concurrency:`, plus `gh-aw`-specific keys:
  * `engine:` — the agent runtime (`copilot`) and `model:` (`claude-sonnet-5`).
  * `network:` — the network policy enforced by a firewall proxy around the agent.
  * `tools:` — which tool servers the agent may use (for example the GitHub MCP server toolsets, and
    the `repo-memory` store).
  * `safe-outputs:` — the only side effects the agent is allowed to produce. The agent cannot post a
    comment or send a notification directly; it emits a structured *safe output* that a trusted,
    non-agent job then acts on.
  * `imports:` — shared fragments pulled in from
    [`.github/workflows/shared`](../../../../.github/workflows/shared/agentic-workflows): step/job
    fragments *and* parameterised prompt fragments (see
    [Shared reusable jobs and prompt fragments](#shared-reusable-jobs-and-prompt-fragments)).
* **Body** is the agent's prompt: the mission, the investigation protocol, the output format, and the
  guardrails (for example, "read at most 10 source files" or "never write to the knowledge base").
  For the two automatic doctors most of the prompt is imported from shared fragments; the body of the
  workflow file itself only holds the workflow-specific additions.

### Source files and compiled lock files

`gh-aw` does not run the `.md` file directly. It **compiles** each `.md` into a standard GitHub Actions
workflow with the `.lock.yml` extension:

| Source (edit this) | Compiled (generated) |
| --- | --- |
| [`ci-doctor.md`](../../../../.github/workflows/ci-doctor.md) | [`ci-doctor.lock.yml`](../../../../.github/workflows/ci-doctor.lock.yml) |
| [`ci-doctor-mq.md`](../../../../.github/workflows/ci-doctor-mq.md) | [`ci-doctor-mq.lock.yml`](../../../../.github/workflows/ci-doctor-mq.lock.yml) |
| [`ci-doctor-post-commit.md`](../../../../.github/workflows/ci-doctor-post-commit.md) | [`ci-doctor-post-commit.lock.yml`](../../../../.github/workflows/ci-doctor-post-commit.lock.yml) |
| [`ci-doctor-remediation.md`](../../../../.github/workflows/ci-doctor-remediation.md) | [`ci-doctor-remediation.lock.yml`](../../../../.github/workflows/ci-doctor-remediation.lock.yml) |

The `.lock.yml` file is what GitHub Actions actually executes. It is **auto-generated — do not edit it
by hand**. After changing a `.md` source (or any imported shared file), regenerate the lock file:

```bash
gh aw compile
```

See [Maintaining the workflows](#maintaining-the-workflows) for details.
The plain YAML [`ci-doctor-mq-readd.yml`](../../../../.github/workflows/ci-doctor-mq-readd.yml)
companion is not compiled by `gh aw`.

## Workflows at a glance

| | **CI Doctor** | **CI Doctor — Merge Queue** | **CI Doctor — Post-Commit** | **MQ Re-add** |
| --- | --- | --- | --- | --- |
| Source | [`ci-doctor.md`](../../../../.github/workflows/ci-doctor.md) | [`ci-doctor-mq.md`](../../../../.github/workflows/ci-doctor-mq.md) | [`ci-doctor-post-commit.md`](../../../../.github/workflows/ci-doctor-post-commit.md) | [`ci-doctor-mq-readd.yml`](../../../../.github/workflows/ci-doctor-mq-readd.yml) |
| Trigger | On demand: `/ci-doctor` comment on a PR | Automatic: a monitored workflow finishes a `merge_group` run with `failure` | Automatic: a monitored workflow finishes a `push` run with `failure` | PR `dequeued`, plus CI Doctor MQ completion fallback |
| Scope | **Every** failed pipeline on the PR head commit | A **single** failed merge-queue run | A **single** failed post-commit run | One PR dequeue with a trusted transient marker |
| Knowledge base | Reads the MQ pattern database (read-only) | **Writes** the MQ pattern database (investigations + patterns) | **Writes** its own post-commit pattern database (investigations + patterns) | None |
| Output | One consolidated PR comment | Teams notification, optional PR comment, failed-job re-run, transient marker | Microsoft Teams notification only | Re-add eligible PR at most once |
| Remediation | Report only | Re-runs failed jobs if still queued; marks transient failures for possible re-add | Report only (never re-runs or re-queues) | Re-adds only for `CI_FAILURE` after marker validation |
| Engine / model | `copilot` / `claude-sonnet-5` | `copilot` / `claude-sonnet-5` | `copilot` / `claude-sonnet-5` | None (deterministic) |

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
   table visible and a separate, collapsed-by-default `<details>` section for each pipeline's
   diagnosis, remedy, and log evidence; otherwise `noop`/`missing_data`. Readers can expand only the
   pipelines they need without scrolling through every log excerpt.

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
    branches: ["gh-readonly-queue/**"]

if: ${{ github.event_name == 'workflow_dispatch' || (github.event.workflow_run.conclusion == 'failure' && github.event.workflow_run.event == 'merge_group') }}
```

`gh aw compile` requires a positive `workflow_run.branches` scope (check `CTR-021`). Merge-queue runs
always execute on GitHub-created `gh-readonly-queue/<base>/pr-<N>-<sha>` branches in this repository,
including for PRs from forks, so `gh-readonly-queue/**` still covers every merge-queue failure.

A `workflow_dispatch` entry (with `run_id` or `link` inputs) is also provided for manual testing.
The run title carries the source merge-queue branch so the re-add workflow can correlate its
completion fallback with the affected PR.

**What it does / benefits**

* **Root-cause analysis** of a single failed merge-queue run: category, failed jobs, key errors,
  explanation, confidence.
* **Knowledge building**: every investigation is written to a persistent, cross-run knowledge base so
  recurring failures accumulate statistics (`count`, `first_seen`, `last_seen`) over time.
* **Escalation**: when the same failure signature is seen **3 or more times within 12 hours**, it sends
  a second, higher-visibility Teams alert.
* **Transient-failure handling**: when a failure is transient the agent calls
  `remediate_transient_failure`. That job resolves the live queue status, re-runs failed jobs if the
  PR remains queued, and posts a bot-authored eligibility marker bound to the current PR head and
  source run. It never re-adds the PR.
* **Separate re-queueing**: [`ci-doctor-mq-readd.yml`](../../../../.github/workflows/ci-doctor-mq-readd.yml)
  handles `pull_request_target: dequeued` and CI Doctor MQ completion. The completion trigger covers the
  race where analysis posts its marker after the dequeue event. The workflow re-adds only if the
  marker is authored by `github-actions[bot]`, matches the current PR head and source merge-group run,
  the latest queue removal reason is `CI_FAILURE`, and the PR is not already queued. It never
  checks out or executes PR code, supports fork PRs, and re-adds a PR at most once.
* **Rerun-pattern proposals**: for a transient failure with a verified `rerun_search_string`
  candidate, the agent calls `propose_rerun_pattern`, which opens a **human-review pull request** against
  `master` adding it to `errors_to_look_for.json`. The workflow rerunner only uses a string after that PR is merged — see
  [Automated re-run integration](#automated-re-run-integration).

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
`notify_teams_recurring`, `add_comment`, `remediate_transient_failure`, `propose_rerun_pattern`, `noop`, or
`missing_data`.

## CI Doctor MQ — Re-add (deterministic)

The plain GitHub Actions workflow
[`ci-doctor-mq-readd.yml`](../../../../.github/workflows/ci-doctor-mq-readd.yml) performs no AI analysis.
It is triggered by a PR `dequeued` event and by completion of CI Doctor MQ. The latter is a bounded
race fallback: if the dequeue trigger arrives before the agent has posted its marker, the analysis
completion starts the same validation again.

The workflow accepts only a bot-authored eligibility comment whose marker matches the PR number and
current head SHA, and whose source run was a failed `merge_group` run for that PR. It also checks that
the latest queue event is a removal, that the marker was created after the latest queue admission,
and that live merge-queue state confirms the PR is out of the queue. For a dequeue event, it uses the
required top-level webhook `reason`; on the completion fallback, it verifies the GraphQL
`RemovedFromMergeQueueEvent.reason` field. Both paths accept only `CI_FAILURE`. Merged, closed, draft,
stale, already-queued, unmarked, or previously re-added PRs are skipped. The workflow checks out only
trusted repository scripts from the default branch and never checks out or executes PR code.
The checks and enqueue operation are implemented in
[`readd_transient_failure.py`](../../../../.github/scripts/agentic-workflows/readd_transient_failure.py).

The dequeue webhook's top-level `reason` is a required string, using the
[documented removal reasons](https://docs.github.com/en/webhooks/webhook-events-and-payloads#pull_request).
Only `CI_FAILURE` is eligible; `CI_TIMEOUT`, manual removals, all other reasons, and unknown values are
skipped. A dequeue event must have its documented top-level reason. The queue timeline is read from
GraphQL because the REST timeline-event contract does not document a removal-reason field. A missing
GraphQL reason is tolerated only on the direct webhook path, where the webhook provides the required
reason; the completion fallback fails closed if GraphQL does not provide `CI_FAILURE`.

## CI Doctor — Post-Commit (automatic)

**What it is** An automatic, always-on investigator that reacts to **post-commit** (push) CI failures,
builds its own persistent knowledge base of failure patterns, and reports each failure to Microsoft
Teams. Unlike the Merge Queue doctor, it **never re-runs or re-queues** any pipeline — its sole purpose
is to collect problems and report them.

**How it is triggered** It runs on the `workflow_run: completed` event of the same fixed list of
monitored workflows as the Merge Queue doctor, and only proceeds when the run was a **push**
(post-commit) run that **failed**:

```yaml
on:
  workflow_run:
    workflows: ["Linux (Ubuntu 22.04, Python 3.11)", "Windows (VS 2022, Python 3.11, Release)", ...]
    types: [completed]

if: ${{ github.event_name == 'workflow_dispatch' || (github.event.workflow_run.conclusion == 'failure' && github.event.workflow_run.event == 'push') }}
```

A `workflow_dispatch` entry (with `run_id` or `link` inputs) is also provided for manual testing.

**What it does / benefits**

* **Root-cause analysis** of a single failed post-commit run: category, failed jobs, key errors,
  explanation, confidence — identical to the Merge Queue analysis.
* **Knowledge building**: every investigation is written to its own persistent, cross-run knowledge base
  (on the `memory/ci-doctor-post-commit` branch, under the `post-commit/` subdirectory) so recurring
  post-commit failures accumulate statistics (`count`, `first_seen`, `last_seen`) over time, kept
  separate from the merge-queue data.
* **Report only**: it emits a single `notify_teams` safe output with `source: "post_commit"` and nothing
  else — no PR comment, no re-run, no re-queue, no recurrence escalation.

**Algorithm** The body follows the same phases as the Merge Queue doctor, minus remediation and
recurrence escalation: trigger detection & triage (verify a failed `push` run), hint-first log analysis,
historical context, root-cause investigation within the source-inspection safeguards, pattern storage
(investigation record + append-only index + per-signature pattern file, all schema-validated), reporting,
and output validation. Its pattern schema drops the `rerun_search_string` field (there is no re-run
integration) and tracks `affected_commits` instead of `affected_prs`. The run always ends by calling
exactly one of `notify_teams` (with `source: "post_commit"`), `noop`, or `missing_data`.

## CI Doctor — Weekly Remediation (automatic)

**What it is** A weekly maintenance workflow that *consumes* the knowledge base the two automatic
doctors build, rather than diagnosing a new failure. It reviews the recurring failures recorded over the
past week, proposes a concrete per-issue remediation, opens draft pull requests for the code-fixable
ones, and uploads a Markdown report covering every identified issue. It is strictly **read-only** with
respect to the knowledge base and never re-runs or re-queues any pipeline.

**How it is triggered** On a weekly `schedule` (fuzzy `weekly on monday`) and on demand via
`workflow_dispatch` (with an optional `days` look-back input, default `7`). There is no `workflow_run`
trigger and no actor allow-list — it is a scheduled maintenance job.

**What it does / benefits**

* **Weekly synthesis**: instead of one failure at a time, it looks at the whole week of accumulated
  patterns across both doctors, groups them by `signature_hash`, and ranks them by reproduction `count`,
  recency, and breadth.
* **Actionable remediation**: for each issue it decides on a concrete fix and classifies it as
  code-fixable or not.
* **Automated fixes**: for the highest-impact code-fixable issues it opens **up to 3 draft pull
  requests** (one issue per PR), each with `akashchi` as reviewer and assignee and the
  `agentic-workflows` + `automated-fix` labels, applied automatically by the workflow. Because
  remediation frequently touches `.github/` CI infrastructure, `create-pull-request` is configured with
  `protected-files: allowed`.
* **Durable record**: every identified issue — code-fixable or not — is written to a Markdown report
  uploaded as a run artifact via the `upload_artifact` safe output.

**Algorithm** The body defines a four-phase protocol: (1) triage and group the pre-downloaded patterns;
(2) decide a per-issue remediation and classify code-fixable vs not; (3) create at most 3 pull requests
for the top code-fixable issues (minimal, root-cause changes; PR body carries the problem, investigation
run URLs, pattern signature/count, and the fix); (4) write and upload the Markdown report. The recent
knowledge base is pre-downloaded before the agent starts by the imported
[`collect-ci-doctor-history.md`](../../../../.github/workflows/shared/agentic-workflows/collect-ci-doctor-history.md)
step (script
[`collect_ci_doctor_history.py`](../../../../.github/scripts/agentic-workflows/collect_ci_doctor_history.py)),
which reads the `memory/ci-doctor-mq` and `memory/ci-doctor-post-commit` branches read-only and writes a
ranked `summary.txt` plus the filtered `patterns/`/`investigations/` under
`/tmp/gh-aw/agent/ci-doctor-remediation/`. The run always ends by calling `upload_artifact` (optionally
with up to 3 × `create_pull_request`), or `noop`/`missing_data`. A `report-failure-as-issue` guard flags
the `missing_safe_outputs` no-op so a silent run is surfaced rather than passing green.

## Shared reusable jobs and prompt fragments

Everything the doctors have in common is factored into
[`.github/workflows/shared/agentic-workflows`](../../../../.github/workflows/shared/agentic-workflows)
and pulled into a workflow via the `imports:` key. Two kinds of shared file live there:

* **Jobs and steps** — `download-failure-logs.md`, `collect-pr-info.md`, `notify-teams*.md`,
  `remediate-transient-failure.md`. Imported `steps:` are prepended to the importing workflow, and
  imported `safe-outputs.jobs:` become callable safe outputs for the agent.
* **Prompt fragments** — `ci-doctor-*.md`. Their Markdown body is the shared part of the agent prompt;
  it is parameterised with an `import-schema` and inlined into the importing workflow's prompt at
  compile time.

### Shared jobs and steps

Each shared job/step `.md` file defines only the job *interface and wiring* — its inputs, `permissions:`, and the
sequence of steps. The actual **logic lives in standalone Python scripts** under
[`.github/scripts/agentic-workflows`](../../../../.github/scripts/agentic-workflows) (one script per job,
plus a shared [`common.py`](../../../../.github/scripts/agentic-workflows/common.py)). A shared job step
sparse-checks-out that scripts directory, sets up Python, and runs its script, for example:

```yaml
- name: Checkout agentic-workflow scripts
  uses: actions/checkout@...
  with:
    sparse-checkout: .github/scripts/agentic-workflows
- name: Set up Python
  uses: actions/setup-python@...
- name: Send Teams notification
  shell: python
  run: |
    export PYTHONPATH=.github/scripts/agentic-workflows/:${PYTHONPATH}
    python .github/scripts/agentic-workflows/notify_teams.py
```

To change what a job *does*, edit its script under `.github/scripts/agentic-workflows/`; to change its
inputs, permissions, or step wiring, edit the shared `.md` and recompile.

| Shared file | Kind | Used by | Purpose |
| --- | --- | --- | --- |
| [`download-failure-logs.md`](../../../../.github/workflows/shared/agentic-workflows/download-failure-logs.md) | Pre-agent step | all | Pre-download failed logs and pre-locate error hints before the agent starts. |
| [`collect-pr-info.md`](../../../../.github/workflows/shared/agentic-workflows/collect-pr-info.md) | Pre-agent step | CI Doctor, MQ | Resolve the pull request under investigation and pre-collect its metadata (`pr-info.json` / `pr-info.txt`). |
| [`collect-ci-doctor-history.md`](../../../../.github/workflows/shared/agentic-workflows/collect-ci-doctor-history.md) | Pre-agent step | Weekly Remediation | Pre-download the recent CI Doctor investigations and patterns from both memory branches (read-only) into `/tmp/gh-aw/agent/ci-doctor-remediation/` with a ranked `summary.txt`. |
| [`notify-teams.md`](../../../../.github/workflows/shared/agentic-workflows/notify-teams.md) | Safe-output job | MQ, Post-Commit | Send the investigation summary to Microsoft Teams; upload the statistics artifact. A `source` input selects the `[MQ]` / `[PC]` badge and the artifact name. |
| [`notify-teams-recurring.md`](../../../../.github/workflows/shared/agentic-workflows/notify-teams-recurring.md) | Safe-output job | MQ | Send a recurring-failure escalation alert to Teams. |
| [`remediate-transient-failure.md`](../../../../.github/workflows/shared/agentic-workflows/remediate-transient-failure.md) | Safe-output job | MQ | For a transient failure, re-run failed jobs if the PR is still queued and post a trusted marker for the deterministic MQ re-add workflow. It never re-adds the PR. |
| [`propose-rerun-pattern.md`](../../../../.github/workflows/shared/agentic-workflows/propose-rerun-pattern.md) | Safe-output job | MQ | Re-validate a `rerun_search_string` candidate and open a human-review PR against `master` appending it to `errors_to_look_for.json`. |

**`download-failure-logs.md`** is a *step* fragment (it has no `on:` trigger). It auto-detects its mode
from the environment: **run mode** (`RUN_ID` set) analyses a single run; **PR mode** (`PR_NUMBER` set)
analyses every failed run on a PR head commit. In both modes it writes job logs, per-job error *hint*
files, and a `summary.txt` under `/tmp/gh-aw/agent/ci-doctor/` so the agent can start from a compact
summary instead of downloading logs itself.

**`notify-teams.md`** defines the `notify-teams` safe-output job. It reads the agent's `notify_teams`
item, renders an Adaptive Card (title with an `[MQ]` / `[PC]` badge, a `Source` fact, facts,
description, and a pattern-database statistics table), POSTs it to the `TEAMS_WEBHOOK_URL`, and uploads
the full statistics JSON/Markdown as a workflow artifact. The agent-supplied `source` input
(`merge_queue` / `post_commit`) selects the badge and the artifact name (`ci-doctor-mq-statistics` /
`ci-doctor-post-commit-statistics`).

**`notify-teams-recurring.md`** defines the `notify-teams-recurring` job, used only when a failure has
recurred ≥ 3 times in 12 hours. It renders a condensed escalation card listing the affected PRs and the
recent failure runs.

**`remediate-transient-failure.md`** defines the `remediate-transient-failure` job for transient
failures. The job resolves the PR's live queue status (`common.merge_queue_status`), re-runs only failed
jobs when the PR is still queued (with a loop guard against a second attempt), and posts a bot-authored
marker bound to the current PR head, source failure run, and analysis run. It never re-adds the PR.
The default `GITHUB_TOKEN` provides `actions: write` for re-runs and `pull-requests: write` for the
marker comment.

**`ci-doctor-mq-readd.yml`** is the deterministic companion workflow. It runs on PR dequeue and on CI
Doctor MQ completion (to handle a marker posted after the dequeue trigger). It only re-adds after
checking the bot-authored current-head marker, matching failed merge-group source run, latest
`CI_FAILURE` queue removal, and live queue state. It skips merged, closed, draft, stale, already
queued, or previously re-added PRs. It uses trusted default-branch scripts only; it never checks out
PR code. `MERGE_QUEUE_TOKEN` performs the enqueue because `GITHUB_TOKEN` cannot re-trigger
`merge_group` checks.

**`propose-rerun-pattern.md`** defines the `propose-rerun-pattern` job
([`propose_rerun_pattern.py`](../../../../.github/scripts/agentic-workflows/propose_rerun_pattern.py)),
the only path by which a new rerun search string can reach the workflow rerunner. It re-validates the
agent's candidate (transient category only; at least 3 words / 15 characters after normalization; not
just generic words, step boilerplate, or a hash; no URLs, which would pin it to one dependency release;
must occur in the supplied verbatim log line, which must contain no CR/LF characters and must
not be a step `##[group]` header) and de-duplicates it against
[`errors_to_look_for.json`](../../../../.github/scripts/workflow_rerun/errors_to_look_for.json) on the
default branch, entries added by other open proposal PRs, and any earlier PR for the same signature (a
closed, unmerged PR counts as a rejection and is never re-proposed). If the candidate passes, it creates
branch `ci-doctor/rerun-pattern/<signature-hash>` from the default branch, appends
`{"error_text": ..., "ticket": 9999999, "category": ...}` to `errors_to_look_for.json`, and opens a PR
with a review checklist, labels, and requested reviewers. Concurrent proposals all
append to the end of the file, so after one is merged the others need a trivial rebase. Skipped proposals are reported in the job summary and do not fail the run. All writes (branch,
commit, PR, labels, reviewers) use the `RERUNNER_DB_UPDATE` secret, so the PR triggers CI normally; the
job's own `GITHUB_TOKEN` is limited to `contents: read`.
If an open proposal's error database cannot be read or parsed, the job fails instead of opening a new
PR with an incomplete overlap check.

### Shared prompt fragments

The Merge Queue and Post-Commit doctors run the *same* investigation protocol; they differ only in the
trigger event, the repo-memory branch/subdirectory and schema directory, the field that scopes a failure
(PR vs commit), the Teams `source` badge, and the extra merge-queue-only remediation tooling. Rather than
maintaining two ~600-line prompts, the common protocol is split into three **parameterised prompt
fragments**. Each declares an
[`import-schema`](https://github.github.com/gh-aw/reference/imports/#import-schema-import-schema); the
importing workflow passes its flavour via `uses:`/`with:`, and `gh aw compile` substitutes
`${{ github.aw.import-inputs.<key> }}` in the fragment's frontmatter **and** body before inlining it:

| Fragment | Content | Inputs | Also wires |
| --- | --- | --- | --- |
| [`ci-doctor-investigation-protocol.md`](../../../../.github/workflows/shared/agentic-workflows/ci-doctor-investigation-protocol.md) | Mission, run context, pre-downloaded log layout, trigger detection, Phases 1–4 (triage, log analysis, historical context, bounded root-cause investigation). | `name`, `kind`, `event`, `slug` | `tools.github` toolsets |
| [`ci-doctor-knowledge-base.md`](../../../../.github/workflows/shared/agentic-workflows/ci-doctor-knowledge-base.md) | Phase 5: schema-validated investigation records, append-only index, read-modify-write pattern records, statistics snapshot. | `slug`, `index_scope_field`, `pattern_scope_field`, `scope_ref` | `tools.repo-memory` on `memory/ci-doctor-<slug>`; `post-steps` uploading the `ci-doctor-<slug>-investigations` artifact |
| [`ci-doctor-reporting.md`](../../../../.github/workflows/shared/agentic-workflows/ci-doctor-reporting.md) | Phases 6–7, `notify_teams` field guidance and Teams description template, common guidelines, mandatory safe-output rule, memory strategy. | `name`, `kind`, `event`, `source`, `slug` | — |

For example, the Merge Queue doctor imports them as:

```yaml
imports:
  - uses: shared/agentic-workflows/ci-doctor-investigation-protocol.md
    with: { name: Merge Queue, kind: merge-queue, event: merge_group, slug: mq }
  - uses: shared/agentic-workflows/ci-doctor-knowledge-base.md
    with: { slug: mq, index_scope_field: pr_number, pattern_scope_field: affected_prs, scope_ref: "the PR URL (or null if no PR)" }
  - uses: shared/agentic-workflows/ci-doctor-reporting.md
    with: { name: Merge Queue, kind: merge-queue, event: merge_group, source: merge_queue, slug: mq }
```

and the Post-Commit doctor passes `slug: post-commit`, `event: push`, `source: post_commit`,
`index_scope_field: commit_sha`, `pattern_scope_field: affected_commits`.

**How the prompt is assembled.** `gh-aw` inlines the body of every frontmatter import *before* the
importing workflow's own body, in import order. The effective prompt of each automatic doctor is
therefore: *investigation protocol → knowledge base → reporting → workflow-specific instructions*.
The shared fragments explicitly point the agent at the trailing
`# CI Failure Doctor — <name>: Workflow-Specific Instructions` section, which extends the shared phases
and takes precedence on conflict. The workflow `.md` bodies hold only those additions:

* [`ci-doctor-mq.md`](../../../../.github/workflows/ci-doctor-mq.md): the pre-collected PR context, the
  `rerun_search_string` pattern field (compute and verify a candidate; no backfill of other records),
  Phase 5.5 recurrence escalation, and the `add_comment` / `notify_teams_recurring` /
  `remediate_transient_failure` / `propose_rerun_pattern` safe-output guidance and
  valid call combinations.
* [`ci-doctor-post-commit.md`](../../../../.github/workflows/ci-doctor-post-commit.md): the
  `master`-only scope, the report-only rule, how PR context may be inferred from the commit, and the
  "exactly one safe output" rule.

Because imported prompt text is inlined at compile time, **editing a `ci-doctor-*.md` fragment requires
recompiling every importing workflow** (`gh aw compile`), unlike edits to a workflow's own body, which
is loaded at runtime. The workflow `name:` is pinned in each workflow's frontmatter so the H1 of the
shared fragment does not affect the GitHub Actions workflow name.

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

CI Doctor — Post-Commit persists its knowledge the same way, on a **separate** dedicated branch
(`memory/ci-doctor-post-commit`) under the `post-commit/` subdirectory, so post-commit failure patterns
never mix with the merge-queue ones. Its records mirror the merge-queue layout but its pattern schema
drops `rerun_search_string` (no re-run integration) and tracks `affected_commits` instead of
`affected_prs`. Its schemas live under
[`.github/ci-doctor-post-commit/schemas`](../../../../.github/ci-doctor-post-commit/schemas).

Every artifact conforms to a committed JSON Schema and is validated immediately after being written:

| Artifact | MQ schema | Post-Commit schema |
| --- | --- | --- |
| Investigation record | [`investigation.schema.json`](../../../../.github/ci-doctor-mq/schemas/investigation.schema.json) | [`investigation.schema.json`](../../../../.github/ci-doctor-post-commit/schemas/investigation.schema.json) |
| Pattern record | [`pattern.schema.json`](../../../../.github/ci-doctor-mq/schemas/pattern.schema.json) | [`pattern.schema.json`](../../../../.github/ci-doctor-post-commit/schemas/pattern.schema.json) |
| Investigations index | [`index.schema.json`](../../../../.github/ci-doctor-mq/schemas/index.schema.json) | [`index.schema.json`](../../../../.github/ci-doctor-post-commit/schemas/index.schema.json) |

CI Doctor (the PR workflow) mounts the merge-queue branch **read-only** to detect known recurring issues,
and never writes to it.

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

The [`workflow_rerun`](../../../../.github/scripts/workflow_rerun) tooling re-runs **every** failed
monitored pipeline whose logs contain a known transient error string, so each string it uses must be
specific. It reads a single, **human-reviewed** source:
[`errors_to_look_for.json`](../../../../.github/scripts/workflow_rerun/errors_to_look_for.json), which only
changes through reviewed PRs to `master`. It never reads the CI Doctor memory branch.

Pattern records in `mq/patterns/` still carry a `rerun_search_string`, but it is only a **candidate**:
for **transient** categories (`Flaky Test`, `Infrastructure`, `Network`, `External Service`) the agent
stores a short, stable substring taken verbatim from a log line in a **failed step within a failed
job**, with both conclusions verified as `failure` and the line mapped to that step. Full failed-job
logs also contain successful steps, and the downloaded job list includes cancelled jobs; neither is
eligible evidence. If either failure conclusion or the line's step provenance cannot be verified, the
candidate is `null`. Eligible candidates are verified against the exact `log_analyzer.py` matcher;
for deterministic categories it also stores `null`. The rerunner does not read this
field. Instead the agent calls `propose_rerun_pattern`, and the
[`propose-rerun-pattern`](#shared-jobs-and-steps) job opens a PR against `master` appending the candidate to
`errors_to_look_for.json` with ticket `9999999` (the reviewer may replace it with a real ticket). A maintainer reviews the string for specificity (it must not match unrelated
failures) and merges or closes the PR; closing it permanently rejects that signature's proposal.

The `rerun_search_string` values that the doctor had written to `mq/patterns/` before this review flow
existed were migrated into `errors_to_look_for.json` (ticket `9999999`), except clearly over-generic ones
(for example step titles, artifact-upload notices, or bare `Unexpected failures`), so no previously
approved behaviour was lost.

### Secrets and permissions

All workflows run with `permissions: read-all` for the agent itself; each *safe-output* job requests
only the narrow permission it needs. The workflows rely on the following secrets:

| Secret | Used by | Purpose |
| --- | --- | --- |
| `TEAMS_WEBHOOK_URL` | `notify-teams` (MQ + Post-Commit), `notify-teams-recurring` | Microsoft Teams incoming webhook. |
| `MERGE_QUEUE_TOKEN` | `ci-doctor-mq-readd.yml` | PAT / App token with `contents: write` + `pull_requests: write` to re-queue a PR (the default token cannot re-trigger `merge_group` runs). |
| `RERUNNER_DB_UPDATE` | `propose-rerun-pattern` | PAT / App token with `contents: write` + `pull_requests: write` to push `ci-doctor/rerun-pattern/*` branches and open review PRs against `master` that edit `errors_to_look_for.json` (unlike `GITHUB_TOKEN`, PRs it opens trigger CI). |
| `GITHUB_TOKEN` | log download, `remediate-transient-failure`, `ci-doctor-mq-readd.yml` | Standard GitHub API access (re-run failed jobs, read live merge-queue status and verify markers). |

## Maintaining the workflows

1. Edit the **`.md` source** (or an imported file under
   [`.github/workflows/shared`](../../../../.github/workflows/shared/agentic-workflows)) — never the
   generated `.lock.yml`.
2. Recompile:

   ```bash
   gh aw compile
   ```

   This regenerates `ci-doctor.lock.yml` / `ci-doctor-mq.lock.yml` / `ci-doctor-post-commit.lock.yml`.
   The lock file carries a hash of the frontmatter and body, so unrelated edits will not always change it.
3. Commit **both** the `.md` and the regenerated `.lock.yml` together.

When changing the shared investigation protocol, edit the relevant
[`ci-doctor-*.md` prompt fragment](#shared-prompt-fragments) once rather than both workflow bodies, keep
any new flavour-dependent wording behind an `import-schema` input, and recompile **both** importing
workflows. Put behaviour that applies to only one doctor in that workflow's own body.

When changing the merge-queue knowledge-base format, update the matching schema under
[`.github/ci-doctor-mq/schemas`](../../../../.github/ci-doctor-mq/schemas) so the in-workflow validation
stays in sync. Likewise, when changing the post-commit knowledge-base format, update the matching schema
under [`.github/ci-doctor-post-commit/schemas`](../../../../.github/ci-doctor-post-commit/schemas).

The repository ships an `ov-agentic-workflows` [agent skill](../../../../.agents/skills/ov-agentic-workflows/SKILL.md)
that captures these editing rules and common tasks. When you work on these workflows with an AI coding
assistant (such as GitHub Copilot), it loads the skill automatically to apply the correct procedure and
guardrails.

## See also

* [Overview of the OpenVINO GitHub Actions CI](./overview.md)
* [Reusable Workflows](./reusable_workflows.md)
* [GitHub Actions security guidelines](./security.md)
* [GitHub Agentic Workflows documentation](https://github.github.com/gh-aw/introduction/overview/)
