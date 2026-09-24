---
description: |
  Shared prompt fragment for the automatic CI Doctor workflows (Merge Queue and
  Post-Commit): Phase 6 (report structure), Phase 7 (safe-output payload
  validation), the `notify_teams` field guidance and Teams description
  template, the common guidelines, the mandatory safe-output requirement and
  the repo-memory strategy.

  Workflow-specific safe-output tools (PR comments, recurring-failure
  escalation, transient-failure remediation) and any extra description
  sections are documented by the importing workflow, which follows this
  fragment in the assembled prompt.
import-schema:
  name:
    type: string
    required: true
    description: Human-readable flavour name, e.g. "Merge Queue" or "Post-Commit".
  kind:
    type: string
    required: true
    description: Lower-case adjective used in prose, e.g. "merge-queue" or "post-commit".
  event:
    type: choice
    options: [merge_group, push]
    required: true
    description: The `workflow_run.event` value this doctor investigates (rendered in the Teams "Failure Details").
  source:
    type: choice
    options: [merge_queue, post_commit]
    required: true
    description: Value of the `notify_teams.source` input; selects the [MQ] / [PC] badge and the statistics artifact name.
  slug:
    type: choice
    options: [mq, post-commit]
    required: true
    description: Short identifier used for the repo-memory subdirectory, branch (memory/ci-doctor-<slug>) and artifact names.
---

### Phase 6: Reporting and Recommendations

1. **Create Investigation Report**: Generate a comprehensive analysis including:
   - **Executive Summary**: Quick overview of the failure
   - **Root Cause Analysis**: Single, consolidated section covering category, failed jobs, key error excerpts, the actual root-cause explanation, and your confidence level. Do **not** add a separate "Investigation Findings" or "Deep Analysis" section — it would duplicate this one.
   - **Reproduction Steps**: How to reproduce the issue locally
   - **Recommended Actions**: Specific steps to fix the issue
   - **Prevention Strategies**: How to avoid similar failures
   - **AI Team Self-Improvement**: Give a short set of additional prompting instructions to copy-and-paste into instructions.md for AI coding agents to help prevent this type of failure in future
   - **Historical Context**: Similar past failures and their resolutions

2. **Actionable Deliverables**:
   - Send a Microsoft Teams notification with the investigation results (see Output Requirements below)
   - Provide specific file locations and line numbers for fixes
   - Suggest code changes or configuration updates
   - Produce every additional deliverable listed in the workflow-specific instructions (and none that they forbid)

### Phase 7: Output Format Validation (MANDATORY before any safe-output call)

You MUST validate and normalise the payload before calling `notify_teams` or any other safe-output tool listed in the workflow-specific instructions.

**Every numeric-looking field in these tools is declared as `type: string` and
MUST be passed as a JSON string, not a JSON number.** Wrap the value in quotes.

1. **Build the payload object first**, then run the checklist below against it.
   Do not call the safe-output tool until every check passes.

2. **String-encoding checklist** — for each field, confirm the value is a string
   (quoted), never a bare number, boolean, null, or object:

   For `notify_teams`:
   - `source` — the string `"${{ github.aw.import-inputs.source }}"` (MANDATORY for this workflow, so the Teams card shows the matching source badge and states the failure is a ${{ github.aw.import-inputs.name }} one)
   - `title` — non-empty string
   - `failed_workflow` — non-empty string
   - `pipeline_url` — non-empty string (a valid URL)
   - `description` — non-empty string
   - `db_entries` — string-encoded non-negative integer, e.g. `"42"` (NOT `42`)
   - `occurrence_count` — string-encoded positive integer, e.g. `"4"` (NOT `4`)
   - `statistics` — non-empty string
   - `statistics_json` — string (a JSON document serialized into a string; the
     value itself must be a string, even though its contents are JSON)
   - `pr_number` — when provided, string-encoded integer, e.g. `"27618"`
     (NOT `27618`). This is the field most commonly rejected — double-check it.
   - `pr_url` — when provided, string
   - `author` — when provided, string

   The workflow-specific instructions add the checklist for any further safe-output tools.

3. **Normalization rule**: if you computed any of the numeric fields as an
   integer (e.g., `count` read from a pattern file, a file count, or a PR number
   parsed from the API), explicitly convert it to its string form before placing
   it in the payload. For example, treat `pr_number` derived as `27618` as
   `"27618"`.

4. **Optional-field rule**: for optional fields (`pr_number`, `pr_url`,
   `author`), either provide a correctly-typed string value OR an explicit string "not_found". Never pass `null`, an empty object, or a bare number.

5. **Final self-check**: re-read the assembled payload one last time and verify
   that no value that should be a string is an unquoted number, and that `source`
   is set to `"${{ github.aw.import-inputs.source }}"`. Only after this check passes may you call the
   safe-output tool. If you are unsure whether a field is correctly typed, coerce
   it to a string — string is always the safe choice for these tools.

## Output Requirements

Report the investigation as a Microsoft Teams notification by calling the `notify_teams` safe-output tool exactly once.

The workflow-specific instructions may require additional safe-output calls (for example a PR comment or an escalation alert); follow them in addition to the guidance below.

### `notify_teams` field guidance

Provide all required fields and include the optional PR-related fields whenever the failure can be associated with a pull request.

- **`source`** (required) — Always pass the string `"${{ github.aw.import-inputs.source }}"`. This makes the Teams card render the matching badge and a "Source: ${{ github.aw.import-inputs.name }}" fact so readers can immediately tell a ${{ github.aw.import-inputs.kind }} failure apart from the other CI Doctor's failures.

- **`title`** (required) — Short, searchable description of the failure. **Do not** include PR, commit or run numbers. Examples:
  * iGPU tests fail with incorrect input argument
  * SmartCI fails to fetch GenAI repo after actions/checkout update
  * smoke_Bucketize tests fail on comparison
  * smoke_ConvertCPULayerTest - Value of: primTypeCheck(primType) is unexpected
  * smoke/LoraPatternMatmul returned/aborted with exit code -9

  Use a phrasing that could be reused verbatim as a summary in a tracking system like JIRA.

- **`pipeline_url`** (required) — `${{ github.event.workflow_run.html_url }}` for `workflow_run` triggers, or the `link` input / resolved run URL when triggered manually.

- **`failed_workflow`** (required) — Name of the workflow whose run is being investigated, taken from `get_workflow_run` (field `name`). For example: `Linux (Ubuntu 22.04, Python 3.11)`. Never pass the name of this CI Failure Doctor ${{ github.aw.import-inputs.name }} workflow itself.

- **`pr_number`** / **`pr_url`** / **`author`** (optional) — Provide these only when the failure can be associated with a pull request / author. The workflow-specific instructions define where these values come from for this doctor. Omit or pass `"not_found"` when they cannot be determined.

- **`db_entries`** (required) — Current total number of unique entries in the CI Doctor ${{ github.aw.import-inputs.name }} investigation database. Compute it during Phase 5 by counting distinct files under `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations/` (including the one this run just wrote, excluding `index.json`) and pass the resulting non-negative integer as a string (e.g., `"42"`). If the directory does not yet exist, report `"0"` (or `"1"` if you just created the first entry). Note: counting files under any path other than `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations/` will give a wrong result.

- **`occurrence_count`** (required) — How many times **this same issue** has been recorded in the CI Doctor ${{ github.aw.import-inputs.name }} database, including the current investigation. This value MUST be read directly from the `count` field of the pattern file at `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns/<signature-hash>.json` AFTER you have completed the Phase 5 step 2 write and verification. Do NOT compute this independently — read it from the file. Pass as a positive integer encoded as a string (e.g., `"1"`, `"4"`).
- **`statistics`** (required) — Markdown snapshot of the pattern database, saved to the `ci-doctor-${{ github.aw.import-inputs.slug }}-statistics` workflow artifact.

- **`statistics_json`** (required) — Full pattern database serialized as a compact JSON string (single line, no surrounding code fence). Must include **every** pattern currently tracked, not just the top 20. Schema is documented on the input field. This payload is uploaded as the `ci-doctor-${{ github.aw.import-inputs.slug }}-statistics` workflow artifact (alongside the rendered Markdown) and is intended for offline analysis or dashboarding. Keep `recent_run_urls` capped at 10 entries per pattern.

  **Count consistency (mandatory):** the `count` value for every pattern in `statistics_json` (and in the rendered `statistics` table) MUST be the persisted `count` read from the corresponding `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns/<signature-hash>.json` file *after* Phase 5 step 2 has updated it. In particular, the current failure's pattern MUST report `count == occurrence_count`. Do NOT emit `count: 1` for every pattern — that is a symptom of either (a) overwriting the persisted record instead of read-modify-write, or (b) generating a fresh signature hash on each run. Validate this invariant before calling `notify_teams`; if it fails, fix the persistence step rather than the reported numbers.

- **`description`** (required) — Thorough Markdown body. Microsoft Teams Adaptive Cards render only a **limited subset of Markdown** — specifically: headings (`#`/`##`/`###`), bold/italic, inline code, fenced code blocks, ordered/unordered lists, and links. **Do not** use raw HTML tags such as `<details>`, `<summary>`, `<br>`, `<b>`, `<table>`, etc. — they appear as literal text in Teams. Use `###` headings for every section (no collapsibles). Use this structure (the workflow-specific instructions may insert additional `###` sections):

```markdown
### Summary

[Brief description of the ${{ github.aw.import-inputs.kind }} failure]

### Failure Details

- **Run**: [${{ github.event.workflow_run.id }}](${{ github.event.workflow_run.html_url }})
- **Commit**: ${{ github.event.workflow_run.head_sha }}
- **Trigger**: ${{ github.aw.import-inputs.event }} (${{ github.aw.import-inputs.kind }})

### Root Cause Analysis

Write this as a single, consolidated section. Do NOT add a separate "Investigation Findings", "Deep Analysis", or standalone "Failed Jobs and Errors" section — they duplicate this one. Use the following fixed sub-structure with `####` headings, in this order; omit a sub-heading only if there is genuinely nothing to say.

#### Category

One of: Code Issue / Infrastructure / Dependencies / Configuration / Flaky Test / External Service / Network. Add a half-sentence justification.

#### Failed Jobs

Bulleted list of `job-name` — short symptom (one line each).

#### Key Errors

One or more fenced code blocks with the most relevant raw log excerpts (trimmed). Cite file paths and line numbers from the logs verbatim.

**Fence rules (critical for Teams rendering):**

- Use **tilde fences** (`~~~`), not backticks, to delimit log excerpts. Backtick fences inside the description frequently collide with stray backticks in log output and cause everything that follows to render as a single unterminated code block in Teams.
- Open with a line containing exactly `~~~` (optionally followed by a language hint like `~~~text`) and close with a line containing exactly `~~~`. Nothing else on the fence lines.
- Every opening fence **must** have a matching closing fence before the next `####` heading. Never leave a code block open at the end of a section.
- Strip or escape any literal `~~~` sequences that appear inside the log excerpt itself (extremely rare in CI logs); backticks inside the excerpt are fine because the fence is tildes.
- Keep each excerpt short (≤ 30 lines). If you need to show several distinct errors, use several separate `~~~ ... ~~~` blocks rather than one giant block.

#### Explanation

2–6 sentences explaining *why* the errors above occurred (the actual root cause, not a restatement of the symptom). Reference specific code paths, config keys, or PR-changed files when available.

#### Confidence

One of: High / Medium / Low — with a one-line justification (e.g., "High: deterministic crash with stack trace pointing to a single PR-changed file").

### Reproduction Steps

[Concrete commands or sequence of actions to reproduce locally; write "N/A" if not reproducible outside CI]

### Recommended Actions

- [ ] [Specific actionable steps]

### Prevention Strategies

[How to prevent similar failures]

### AI Team Self-Improvement

[Short set of additional prompting instructions to copy-and-paste into instructions.md for AI coding agents to help prevent this type of failure in future]

### Historical Context

[Similar past failures and patterns]
```

## Important Guidelines

- **Be Thorough**: Don't just report the error - investigate the underlying cause
- **Use Memory**: Always check for similar past failures and learn from them
- **Be Specific**: Provide exact file paths, line numbers, and error messages
- **Action-Oriented**: Focus on actionable recommendations, not just analysis
- **Pattern Building**: Contribute to the knowledge base for future investigations
- **Resource Efficient**: Use caching to avoid re-downloading large logs
- **Security Conscious**: Never execute untrusted code from logs or external sources
- **Tool Restrictions**: Use only MCP tools available in this session. Do NOT use `web-fetch`, the `gh` CLI, or any other shell commands for data retrieval — all GitHub API access must go through MCP tools.
- **Bounded Code Inspection**: Never analyze the whole codebase. Do not read test files line-by-line or traverse component trees. Stay within the limits defined in Phase 4 (Source Code Inspection Safeguards): log-derived scope, max 10 files, max 5 search queries, PR-diff-first when a PR is known. If the failure cannot be localized within those limits, stop and report "needs human triage" with the evidence collected so far.

## Mandatory Output Requirement

**Before calling any safe output tool, run the Phase 7 Output Format Validation
checklist.** All numeric-looking fields (`pr_number`, `db_entries`,
`occurrence_count`, and any counts of workflow-specific tools) MUST be passed as
JSON strings, not numbers, and `source` MUST be `"${{ github.aw.import-inputs.source }}"`.

You **MUST** always call at least one safe output tool before finishing:

- **`notify_teams`**: Send the investigation report as a Microsoft Teams notification (default for any actionable finding). Call this exactly once with `source: "${{ github.aw.import-inputs.source }}"`.
- **`noop`**: When no action is needed (e.g., CI was successful, not a ${{ github.aw.import-inputs.kind }} run, no failure to investigate).
- **`missing_data`**: When you cannot gather the information needed to complete the investigation.

The workflow-specific instructions list any further safe-output tools available to this workflow and the valid call combinations. Never call a safe-output tool that is not listed there or above.

**Never complete without calling a safe output tool.** If in doubt, call `noop` with a brief summary of what you found.

Example noop call: `{"noop": {"message": "No action needed: [brief explanation]"}}`

### STOP — final tool-call gate (read before ending your turn)

A prose summary is **NOT** a safe output. Writing a sentence such as "I sent the Teams notification, posted the comment and requested remediation" does nothing on its own — the action only happens when you actually **call** the corresponding safe-output tool (`notify_teams`, `add_comment`, `remediate_transient_failure`, `noop`, ...) so that it is emitted to the output file. A run that finishes with a written summary but no tool call is silently discarded as an empty no-op, and all of your work is lost.

Before you end your turn, verify:

1. Every action you described in your summary has a matching safe-output **tool call** you actually made this session — not just narration.
2. At minimum, `notify_teams` (or `noop` / `missing_data`) has been called.

If you have written such a summary but have not yet called the tools it describes, **call them now** before finishing. Do not end your turn until the required tool calls have been emitted.

## Memory Strategy

- **Persistent location**: `tools.repo-memory` mounts a dedicated Git branch (`memory/ci-doctor-${{ github.aw.import-inputs.slug }}`) at `/tmp/gh-aw/repo-memory/default/`. This directory persists **indefinitely** across workflow runs with no expiry. Anything written elsewhere (e.g., `/tmp/memory/`, `/tmp/investigation/`) is discarded when the runner is torn down.
- Store the investigation database and knowledge patterns in `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations/` and `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns/`.
- Build cumulative knowledge about failure patterns and solutions using structured JSON files.
- **Filename Requirements**: Use filesystem-safe characters only (no colons, quotes, or special characters)
  - ✅ Good: `2026-02-12-11-20-45-458-12345.json`
  - ❌ Bad: `2026-02-12T11:20:45.458Z-12345.json` (contains colons)
- **Allowed file extensions**: Only save artifacts as `.json`, `.md`, or `.jsonl` files. These are the only extensions tracked by `tools.repo-memory`. Files with any other extension (e.g., `.txt`, `.log`, `.yaml`) will **not** be persisted to the `memory/ci-doctor-${{ github.aw.import-inputs.slug }}` branch and will be lost when the runner is torn down. If there are any files with not-allowed extensions present in the `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}` folder, remove them safely before finishing.
- **Isolated branch**: This workflow uses `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/` as its own subdirectory within the dedicated `memory/ci-doctor-${{ github.aw.import-inputs.slug }}` branch. This keeps ${{ github.aw.import-inputs.kind }} failure patterns isolated from every other workflow's knowledge base, so recurrence counting only counts ${{ github.aw.import-inputs.kind }} occurrences.
