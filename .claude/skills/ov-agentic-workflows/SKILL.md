---
name: ov-agentic-workflows
description: Author, edit, compile, validate, and debug the OpenVINO gh-aw agentic CI workflows (ci-doctor, ci-doctor-mq) and their shared safe-output jobs. Use when a user wants to change an agentic workflow's prompt, triggers, tools, permissions, or safe-outputs; add or edit a shared job/step under .github/workflows/shared/agentic-workflows; recompile the .lock.yml; update the ci-doctor-mq JSON schemas; or test a workflow via workflow_dispatch. Do NOT use for regular (non-agentic) GitHub Actions workflows, plugin/runtime code, or diagnosing a specific live CI failure.
---

# Work on OpenVINO Agentic Workflows

Guides changes to the repository's `gh-aw` agentic workflows and their shared jobs.

**Read [docs/dev/ci/github_actions/agentic_workflows.md](../../../docs/dev/ci/github_actions/agentic_workflows.md) first** — it is the canonical description of what these workflows are, how they are built, their algorithms, and the shared jobs. This skill covers the *how-to* of editing them safely; do not duplicate the doc's content, reference it.

For the framework itself, consult the official [GitHub Agentic Workflows (`gh-aw`) documentation](https://github.github.com/gh-aw/introduction/overview/) — the authoritative reference for frontmatter keys, `imports`, `safe-outputs`, tools, and the `gh aw compile` workflow.

## Key files

| Purpose | Path |
| --- | --- |
| PR investigator (source) | `.github/workflows/ci-doctor.md` |
| Merge-queue investigator (source) | `.github/workflows/ci-doctor-mq.md` |
| Compiled workflows (generated) | `.github/workflows/ci-doctor*.lock.yml` |
| Shared steps / safe-output jobs | `.github/workflows/shared/agentic-workflows/*.md` |
| Knowledge-base schemas | `.github/ci-doctor-mq/schemas/*.schema.json` |
| Re-runner matcher (integration) | `.github/scripts/workflow_rerun/log_analyzer.py` |

## Golden rules

1. **Never edit a `.lock.yml` by hand.** It is generated. Edit the `.md` source (or an imported shared file), then recompile.
2. **Always recompile after editing** any `.md` source or imported file:
   ```bash
   gh aw compile
   ```
   Commit the `.md` **and** the regenerated `.lock.yml` together in the same change.
3. **Keep the two `if:` / `on:` guards intact.** `ci-doctor` is gated to an allow-list of actors; `ci-doctor-mq` only proceeds for failed `merge_group` runs. Do not loosen these without explicit intent.
4. **Preserve the source-inspection safeguards** (≤ 10 files, ≤ 5 searches, PR-diff first) when editing the investigation prompt — they bound cost and blast radius.

## Common tasks

### Edit a workflow's prompt, triggers, tools, or permissions
1. Edit the frontmatter or body of the relevant `.md` source.
2. If you touch `on:`/`if:`, re-read the guard logic in the doc's *How it is invoked / triggered* sections to keep the semantics correct.
3. Recompile and commit both files.

### Add or change a shared safe-output job or step
1. Edit the file under `.github/workflows/shared/agentic-workflows/`. A *step* fragment has no `on:` and is prepended to importers; a *safe-output job* lives under `safe-outputs.jobs:`.
2. Ensure the consuming workflow lists the file under `imports:` and that its body instructs the agent when to call the new safe output.
3. Give each safe-output job the **narrowest** `permissions:` it needs.
4. Recompile every workflow that imports the file.

### Add a new safe output
- Define the job (inputs, `permissions`, steps) in a shared file, import it, and document in the workflow body exactly when the agent may emit it and the valid combinations. Remember: **all numeric-looking safe-output fields must be passed as strings** (see pitfalls).

### Change the knowledge-base format (ci-doctor-mq)
1. Update the matching schema under `.github/ci-doctor-mq/schemas/` (`investigation`, `pattern`, or `index`).
2. Update the Phase 5 write/validation instructions in `ci-doctor-mq.md` so the agent writes conforming artifacts.
3. Validate a sample artifact against the schema:
   ```bash
   python3 -c "import jsonschema, json, sys; jsonschema.validate(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))" sample.json .github/ci-doctor-mq/schemas/pattern.schema.json
   ```

## Pitfalls to check before finishing

- **Lock file out of date** — did you run `gh aw compile` and stage the `.lock.yml`?
- **Safe-output typing** — every numeric-looking field (`pr_number`, `db_entries`, `occurrence_count`, `recent_count`) must be a quoted string, never a bare number.
- **Teams markdown** — Adaptive Cards render a limited subset; use `~~~` (tilde) fences for log excerpts, no raw HTML (`<details>`, `<br>`, `<table>`).
- **Signature hashing** — the pattern hash must stay job-agnostic (normalized error + category, no job name), or recurrence counting breaks.
- **Read-only knowledge base** — `ci-doctor` must never write under `/tmp/gh-aw/repo-memory/default/`; only `ci-doctor-mq` writes.
- **Secrets** — new Teams/queue behavior may need `TEAMS_WEBHOOK_URL` / `MERGE_QUEUE_TOKEN`; the default `GITHUB_TOKEN` cannot re-trigger `merge_group` runs.
