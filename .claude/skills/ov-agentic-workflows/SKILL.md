---
name: ov-agentic-workflows
description: Author, edit, compile, validate, and debug any OpenVINO gh-aw agentic workflow (e.g. ci-doctor, ci-doctor-mq, and any future ones) and their shared safe-output jobs under .github/workflows/shared/agentic-workflows. Use when a user wants to create a new agentic workflow; change an existing one's prompt, triggers, tools, permissions, or safe-outputs; add or edit a shared job/step; recompile a .lock.yml; update a workflow's persisted-state JSON schemas; or test a workflow via workflow_dispatch. Do NOT use for regular (non-agentic) GitHub Actions workflows, plugin/runtime code, or diagnosing a specific live CI failure.
---

# Work on OpenVINO Agentic Workflows

Guides changes to **any** `gh-aw` agentic workflow in this repository and its shared jobs — not just
the ones that exist today. New agentic workflows are expected to be added over time; treat every
`.github/workflows/*.md` file with `gh-aw` frontmatter (`on:`, `engine:`, `safe-outputs:`, etc.) as in
scope for this skill.

**Read [docs/dev/ci/github_actions/agentic_workflows.md](../../../docs/dev/ci/github_actions/agentic_workflows.md) first** — it documents the workflows that exist today (`ci-doctor`, `ci-doctor-mq`) in
detail: what they are, how they are built, their algorithms, and the shared jobs. Treat it as the
worked example of the concepts in this skill, and keep it up to date (see the **Skill
self-improvement** section below) whenever a new agentic workflow is added or an existing one changes
meaningfully.

For the framework itself, consult the official [GitHub Agentic Workflows (`gh-aw`) documentation](https://github.github.com/gh-aw/introduction/overview/) — the authoritative reference for frontmatter
keys, `imports`, `safe-outputs`, tools, and the `gh aw compile` workflow.

## Identifying agentic workflows

* **Source files**: `.github/workflows/*.md` with a `gh-aw` YAML frontmatter (`on:`, `engine:`,
  `safe-outputs:`, `tools:`, ...) — as opposed to a plain `.yml` workflow.
* **Compiled output**: each source `<name>.md` compiles to a generated `<name>.lock.yml` in the same
  directory. This is what GitHub Actions actually runs.
* **Shared building blocks**: step/job fragments imported by one or more agentic workflows live under
  `.github/workflows/shared/agentic-workflows/*.md`.
* **Known examples today**: `ci-doctor.md` (on-demand PR investigator) and `ci-doctor-mq.md` (automatic
  merge-queue investigator). Do not assume this is the complete list — check `.github/workflows/*.md`
  for the current set of `gh-aw` sources, since more will be added over time.

## Golden rules

1. **Never edit a `.lock.yml` by hand.** It is generated. Edit the `.md` source (or an imported shared
   file), then recompile.
2. **Always recompile after editing** any `.md` source or imported file:
   ```bash
   gh aw compile
   ```
   Commit the `.md` **and** the regenerated `.lock.yml` together in the same change.
3. **Keep trigger guards intact.** Agentic workflows commonly gate execution with an `if:` condition
   (an actor allow-list, a specific event/conclusion combination, a slash command, etc.). Do not loosen
   a guard without explicit intent — re-read the workflow's own description of how it is
   invoked/triggered first.
4. **Preserve source-inspection safeguards.** Investigative workflows typically cap how much of the
   codebase the agent may read (file/search budgets, PR-diff-first, "stop and report" conditions). Keep
   these bounds when editing a prompt — they bound cost and blast radius.
5. **Respect `safe-outputs` as the only side-effect boundary.** The agent itself must never be granted
   permission to directly comment, notify, merge, or otherwise mutate shared state — that must go
   through a `safe-outputs` job with its own narrowly-scoped `permissions:`.

## Common tasks

### Create a new agentic workflow
1. Author `.github/workflows/<name>.md` with `gh-aw` frontmatter (`on:`, `permissions:`, `engine:`,
   `network:`, `tools:`, `safe-outputs:`, `imports:` as needed) and a natural-language body describing
   the mission and investigation/action protocol.
2. Reuse existing shared jobs/steps under `.github/workflows/shared/agentic-workflows/` instead of
   duplicating logic (for example log pre-download, notification jobs) — add a new shared file only for
   genuinely new, reusable behavior.
3. Give the workflow the narrowest `permissions:` and the smallest `safe-outputs` surface it needs.
4. Recompile (`gh aw compile`) and commit the `.md` and the generated `.lock.yml` together.
5. Extend [docs/dev/ci/github_actions/agentic_workflows.md](../../../docs/dev/ci/github_actions/agentic_workflows.md) with the new workflow, and apply
   the **Skill self-improvement** guidance below.

### Edit a workflow's prompt, triggers, tools, or permissions
1. Edit the frontmatter or body of the relevant `.md` source.
2. If you touch `on:`/`if:`, re-read that workflow's own "how it is invoked/triggered" description
   (in the doc, or in the workflow's frontmatter comment) to keep the guard semantics correct.
3. Recompile and commit both files.

### Add or change a shared safe-output job or step
1. Edit the file under `.github/workflows/shared/agentic-workflows/`. A *step* fragment has no `on:`
   and is prepended to importers; a *safe-output job* lives under `safe-outputs.jobs:`.
2. Ensure every consuming workflow lists the file under `imports:` and that its body instructs the
   agent when to call the new safe output.
3. Give each safe-output job the **narrowest** `permissions:` it needs.
4. Recompile every workflow that imports the file.

### Add a new safe output
- Define the job (inputs, `permissions`, steps) in a shared file (or inline if genuinely single-use),
  import it, and document in the workflow body exactly when the agent may emit it and any valid
  combinations with other safe outputs. Remember: **all numeric-looking safe-output fields must be
  passed as strings** (see pitfalls).

### Change a workflow's persisted-state / schema format
Some agentic workflows persist structured JSON state across runs (for example via the `repo-memory`
tool). If a workflow you're changing does this:
1. Update the matching JSON Schema file (typically under a workflow-specific directory such as
   `.github/ci-doctor-mq/schemas/`).
2. Update the write/validation instructions in the workflow's `.md` body so the agent writes
   conforming artifacts.
3. Validate a sample artifact against the schema, e.g.:
   ```bash
   python3 -c "import jsonschema, json, sys; jsonschema.validate(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))" sample.json path/to/schema.json
   ```

## Pitfalls to check before finishing

General, applies to any agentic workflow:
- **Lock file out of date** — did you run `gh aw compile` and stage the `.lock.yml`?
- **Safe-output typing** — every numeric-looking field (e.g. `pr_number`, counts) must be a quoted
  string, never a bare number, unless the field is explicitly typed otherwise.
- **Permission creep** — did you give the agent job or a safe-output job broader `permissions:` than it
  needs?
- **Guard regressions** — does the workflow still only run for the intended trigger/actor/event after
  your change?

Known pitfalls from existing workflows (check whether they still apply, and add new ones as you find
them — see below):
- **Teams markdown** (`ci-doctor-mq`) — Adaptive Cards render a limited subset; use `~~~` (tilde)
  fences for log excerpts, no raw HTML (`<details>`, `<br>`, `<table>`).
- **Signature hashing** (`ci-doctor-mq`) — the pattern hash must stay job-agnostic (normalized error +
  category, no job name), or recurrence counting breaks.
- **Read-only knowledge base** (`ci-doctor`) — must never write under `/tmp/gh-aw/repo-memory/default/`;
  only `ci-doctor-mq` writes to it.
- **Secrets** — new Teams/queue-style behavior may need dedicated secrets (e.g. `TEAMS_WEBHOOK_URL`,
  `MERGE_QUEUE_TOKEN`); the default `GITHUB_TOKEN` cannot re-trigger `merge_group` runs.

## Skill self-improvement

This skill must evolve alongside the agentic workflows it describes. Whenever you make a change that
reveals a new rule, reusable pattern, shared job, or footgun, **update this file** (and its mirror,
see below) before finishing:

- **New agentic workflow added** — update the "Known examples today" note, and make sure
  [docs/dev/ci/github_actions/agentic_workflows.md](../../../docs/dev/ci/github_actions/agentic_workflows.md) documents it.
- **New shared job/step introduced** — mention it under "Common tasks" if it changes how a common task
  should be done (e.g. a new reusable notification or remediation pattern).
- **New footgun discovered** (a safe-output typing quirk, a rendering limitation, a required
  permission/secret, a validation step that was missed) — add it to "Pitfalls to check before
  finishing".
- **A rule becomes obsolete** (a workflow is removed, a schema path changes, a guard is redesigned) —
  update or remove the corresponding entry instead of leaving stale guidance in place.
- **Keep it concise** — prefer linking to the (now-updated) doc over duplicating detail here; this
  skill should stay a short, actionable checklist, not a second copy of the documentation.
- **Keep mirrors in sync** — this skill is duplicated under `.claude/skills/ov-agentic-workflows/SKILL.md`
  for tool compatibility. When you edit one copy, apply the identical edit to the other so they never
  drift apart.
