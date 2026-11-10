---
description: |
  Shared prompt fragment for the automatic CI Doctor workflows (Merge Queue and
  Post-Commit): Phase 5 — the persistent knowledge base kept on a dedicated
  repo-memory branch (investigation records, rolling-retention investigations
  index, per-signature pattern records, statistics snapshot). The permanent
  recurrence signal lives in the pattern records; bulky per-run investigation
  records and the index are pruned to rolling windows (Phase 5, step 5) so
  persistence stays under the repo-memory caps below.

  Besides the prompt text this fragment also wires the `repo-memory` tool onto
  the `memory/ci-doctor-<slug>` branch and uploads the investigations/patterns
  directories as a run artifact (`ci-doctor-<slug>-investigations`).

  The JSON artefacts must conform to the schemas committed under
  `.github/ci-doctor-<slug>/schemas/`. The two doctors' schemas differ only in
  the field that scopes a failure (merge-queue: PR; post-commit: commit), which
  is why the scope field names are parameters here.
import-schema:
  slug:
    type: choice
    options: [mq, post-commit]
    required: true
    description: Short identifier used for the repo-memory subdirectory, branch (memory/ci-doctor-<slug>), schema directory (.github/ci-doctor-<slug>/schemas) and artifact name.
  index_scope_field:
    type: choice
    options: [pr_number, commit_sha]
    required: true
    description: Name of the nullable string field in an investigations-index entry that scopes the failure (see index.schema.json).
  pattern_scope_field:
    type: choice
    options: [affected_prs, affected_commits]
    required: true
    description: Name of the capped string array in a pattern record that lists what this signature affected (see pattern.schema.json).
  scope_ref:
    type: string
    required: true
    description: Prose describing the value stored in the pattern scope array for the current run, e.g. "the PR URL (or null if no PR)".
tools:
  repo-memory:
    branch-name: memory/ci-doctor-${{ github.aw.import-inputs.slug }}
    allowed-extensions: [".md", ".json", ".jsonl"]
    # Hard caps enforced by repo-memory on every push; once exceeded the whole
    # push is rejected and the doctor silently stops recording history. The
    # Phase 5 retention/compaction step (step 5) keeps the working set safely
    # under them: per-signature pattern records hold the permanent recurrence
    # signal, while per-run investigation records and the index are pruned to
    # rolling windows (INVESTIGATION_RETENTION_MAX / INDEX_RETENTION_MAX).
    max-file-size: 1048576 # 1MB max — index stays well under this via its rolling window
    max-patch-size: 1048576 # 1MB max
    max-file-count: 1000 # max allowed; pattern records + capped investigation-record window + index + stats
post-steps:
  - name: Upload CI Doctor investigations and patterns (${{ github.aw.import-inputs.slug }})
    if: always()
    uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a  # v7.0.1
    with:
      name: ci-doctor-${{ github.aw.import-inputs.slug }}-investigations
      path: |
        /tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations
        /tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns
      if-no-files-found: ignore
      retention-days: 90
---

### Phase 5: Pattern Storage and Knowledge Building

**Artefact schemas (MANDATORY — read before writing anything):**
Every JSON artefact this phase writes MUST conform to a fixed JSON Schema committed in the repository. The repository is sparse-checked-out at the path reported as **workspace** in the Current Context (environment variable `GITHUB_WORKSPACE`). The schemas are:

- **Investigation records** → `${GITHUB_WORKSPACE}/.github/ci-doctor-${{ github.aw.import-inputs.slug }}/schemas/investigation.schema.json`
  Applies to every `<timestamp>-<run-id>.json` file under `investigations/` **except** the aggregate `index.json`.
- **Pattern records** → `${GITHUB_WORKSPACE}/.github/ci-doctor-${{ github.aw.import-inputs.slug }}/schemas/pattern.schema.json`
  Applies to every `<signature-hash>.json` file under `patterns/`.
- **Investigations index** → `${GITHUB_WORKSPACE}/.github/ci-doctor-${{ github.aw.import-inputs.slug }}/schemas/index.schema.json`
  Applies to the single aggregate `investigations/index.json` file.

Rules that apply to all artefact types:

- Read the relevant schema file **before** composing an artefact so the structure and field names match exactly. Do not invent your own field names or layout.
- Set `"schema_version": "1.0"` on every artefact you write.
- These schemas declare `"additionalProperties": false`. Do **not** add fields that are not defined in the schema — extra fields make the artefact invalid.
- Use the exact field names, types, and `enum` values from the schema. `category` must be one of the seven categories; `confidence` must be `High`/`Medium`/`Low`.
- Timestamp **values** inside JSON use full ISO 8601 with colons (e.g., `2026-05-12T14:30:00Z`); only **file names** use the colon-free `YYYY-MM-DD-HH-MM-SS-sss` form.
- The workflow-specific instructions may require additional schema fields (for example a rerun hook) — apply them on top of the procedures below.

**Validation procedure (run immediately after writing each artefact — MANDATORY):**

1. Read the artefact back from disk and parse it as JSON (this also confirms it is well-formed).
2. Read the matching schema file.
3. Validate the parsed object against the schema. If a JSON Schema validator is available in the run environment (e.g., Python with the `jsonschema` package — `python3 -c "import jsonschema, json, sys; jsonschema.validate(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))" <artefact> <schema>`), use it. Otherwise perform an explicit conformance check covering: every `required` field present; each field's `type`/`enum`/`format` honoured; **no** field outside the schema's `properties` (because `additionalProperties` is `false`); and array `minItems`/`maxItems` limits respected.
4. If validation fails, fix the artefact and repeat until it validates. **Never leave an invalid artefact on disk**, and do not proceed to the next phase with an unvalidated artefact.

1. **Store Investigation**: Save structured investigation data to files in the persistent repo-memory directory:
   - **Persistent path**: `/tmp/gh-aw/repo-memory/default/` is the directory mounted by `tools.repo-memory` and persisted indefinitely via a dedicated Git branch (`memory/ci-doctor-${{ github.aw.import-inputs.slug }}`). Files written here survive across runs permanently. Files written elsewhere are **not** persisted and will be lost.
   - **Workflow-specific subdirectory**: This workflow uses `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/` to keep its investigations isolated from any other workflows using repo-memory.
   - Create the subdirectory if needed: `mkdir -p /tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations /tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns`.
   - Write the investigation report to `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations/<timestamp>-<run-id>.json`
     - The file content MUST conform to the **investigation schema** (`investigation.schema.json`) described in the "Artefact schemas" block above, and MUST be validated with the validation procedure right after writing.
     - **Important**: Use filesystem-safe timestamp format `YYYY-MM-DD-HH-MM-SS-sss` (e.g., `2026-02-12-11-20-45-458`)
     - **Do NOT use** ISO 8601 format with colons (e.g., `2026-02-12T11:20:45.458Z`) - colons are not safe in filenames
   - Store error patterns in `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns/` as `.json` files (one file per failure signature, e.g., `<signature-hash>.json`), each conforming to the **pattern schema** (`pattern.schema.json`)
   - Update the investigations index at `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations/index.json` following the **MANDATORY read-modify-write procedure** in step 1a below (it appends the current run and preserves existing entries; the only sanctioned place the index ever shrinks is the retention prune in step 5). Never recreate this file from scratch.

1a. **Update Investigations Index — MANDATORY read-modify-write procedure**:

   The index at `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations/index.json` is a single rolling-retention aggregate that references the most recent `INDEX_RETENTION_MAX` investigations (older references are trimmed by the retention step, step 5 — the *permanent* recurrence signal lives in the per-signature pattern records, not here). It MUST conform to the **index schema** (`index.schema.json`). Accidentally losing or overwriting entries **during this read-modify-write** is a **critical data-loss bug** — the following procedure exists specifically to prevent it, and you MUST follow it exactly. Deliberate trimming of the oldest entries happens ONLY in step 5, never here.

   **Step A — Read the existing index (never skip):**
   - Attempt to read `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations/index.json`.
   - If it exists, parse it as JSON into `existing`. If it exists but fails to parse (corrupted/truncated), **do NOT overwrite it**: copy it aside to `index.corrupt-<timestamp>.json`, then reconstruct `existing.entries` by scanning every `*.json` investigation record already present under `investigations/` (excluding `index.json` itself) so no prior investigation is dropped.
   - If the file does NOT exist, set `existing = { "schema_version": "1.0", "total": 0, "entries": [] }`.
   - Normalize legacy shapes: if `existing` has a deprecated `investigations` array, merge its elements into `existing.entries` (deduplicating by `investigation_id`+`run_id`) and drop the `investigations` key. Map any legacy `id` field to `investigation_id`.
   - Record `PREV_COUNT = length(existing.entries)`.

   **Step B — Append the current investigation (never remove or replace prior entries):**
   - Build the new entry from the investigation you just wrote, using the fields defined in `index.schema.json`: `investigation_id`, `run_id` (string), `timestamp`, `title`, `category`, `signature_hash`, and `${{ github.aw.import-inputs.index_scope_field }}` (string or null).
   - If an entry with the same `investigation_id` already exists, update that one entry in place; otherwise **append** the new entry to the end of `existing.entries`.
   - Under no circumstances truncate, replace wholesale, reorder-destructively, or shrink `existing.entries` **in this step**. The only allowed mutations here are: appending a new entry, or updating a single matching existing entry in place. (Bounded pruning of the oldest entries happens later, in the retention step 5.)

   **Step C — Recompute and write:**
   - Set `total = length(entries)`.
   - Assert the **in-run never-shrink invariant**: `total >= PREV_COUNT` (this step only appends/updates; it must not drop entries). If this assertion fails, you have a bug — stop, re-read the existing file, and redo from Step A. Do NOT write a smaller index **here** — the retention step 5 is the only place the index is deliberately trimmed.
   - Write the object `{ schema_version: "1.0", total, entries }` back to `index.json`, overwriting the file with the **superset** you just computed.

   **Step D — MANDATORY verification (read-back check):**
   - Read `index.json` back, parse it, and validate against `index.schema.json` (see the validation procedure in the "Artefact schemas" block).
   - Verify `total == length(entries)` and `total >= PREV_COUNT`.
   - Verify the current investigation's `investigation_id` is present in `entries` exactly once.
   - Verify every entry that was in the pre-write `existing.entries` is still present (no prior entry was dropped).
   - If any check fails, **do not leave the shrunken/invalid index on disk** — restore from the pre-write copy and redo from Step A.

   **Common failure modes to avoid:**
   - Recreating `index.json` from scratch (e.g., writing only the current entry) — this destroys all history.
   - Skipping Step A and overwriting instead of appending.
   - Writing a `total` smaller than the `PREV_COUNT` you read in Step A of **this** step (deliberate retention trimming happens only in step 5, never during the append).
   - Dropping the deprecated `investigations` array's contents instead of merging them into `entries`.

2. **Update Pattern Database — MANDATORY read-modify-write procedure**:

   Each failure signature gets exactly one JSON file at `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns/<signature-hash>.json`.

   **Schema:** the authoritative definition is `${GITHUB_WORKSPACE}/.github/ci-doctor-${{ github.aw.import-inputs.slug }}/schemas/pattern.schema.json`. Read that file for the exact field list, types, and constraints, and validate the record against it (see the validation procedure in the "Artefact schemas" block above).

   **Step-by-step procedure (follow EXACTLY in this order):**

   **Step A — Compute signature hash:**
   Derive a stable `<signature-hash>` from ONLY inputs that do NOT change between reruns of the same failure **and that do NOT depend on which job the error occurred in** — the same underlying error frequently surfaces in several different jobs (e.g., the same test failing on Linux and Windows, or across shards), and those MUST collapse into a single pattern:
   - Normalized primary error message: strip absolute paths, line/column numbers, hex addresses, PIDs, timestamps, run IDs, commit SHAs, tmp dirs, UUIDs, and any embedded job / runner / OS / shard names or indices
   - Failure category — MUST be exactly one of the values from the `category` `enum` defined in `pattern.schema.json` (identical to the `category` enum in `investigation.schema.json`). Use the schema's spelling verbatim (e.g., `Code Issue`, `Flaky Test`, `External Service`, `Network`); do **not** invent a category or use the looser prose labels from Phase 4.

   Do **NOT** include the failed job name in the hash. Keying on the job name would split one underlying error into a separate pattern for every job that hits it, inflating the database and breaking recurrence counting. Treat the job name(s) as descriptive metadata only (record them in `title` / the investigation, not in the hash).

   Concatenate the two inputs as `<normalized-error>|<category>`, then compute a hash (e.g., first 16 chars of SHA-256). The same normalized error in the same category MUST always produce the same hash regardless of which job(s) it occurred in, and two reruns of the same failure MUST produce the same hash.

   **Step B — Read existing file:**
   Attempt to read `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns/<signature-hash>.json`.
   - If the file exists, parse it as JSON into a variable called `existing`.
   - If the file does NOT exist, set `existing = null`.

   **Step C — Compute the updated record:**

   ~~~pseudocode
   NOW = current UTC time in ISO 8601 format (e.g., "2026-05-12T14:30:00Z")
   CURRENT_RUN_URL = the URL of the current workflow run
   SCOPE_REF = ${{ github.aw.import-inputs.scope_ref }}

   IF existing != null:
       record.schema_version = "1.0"
       record.signature   = existing.signature
       record.signature_hash = existing.signature_hash
       record.title       = title from investigation (refresh always)
       record.category    = category from investigation (refresh always)
       record.count       = existing.count + 1          ← MUST increment
       record.first_seen  = existing.first_seen         ← NEVER change
       record.last_seen   = NOW
       record.recent_run_urls = [CURRENT_RUN_URL] + existing.recent_run_urls
           → deduplicate by URL, then truncate to first 10 entries
       record.${{ github.aw.import-inputs.pattern_scope_field }} = (if SCOPE_REF: [SCOPE_REF] + existing.${{ github.aw.import-inputs.pattern_scope_field }} else existing.${{ github.aw.import-inputs.pattern_scope_field }})
           → deduplicate, then truncate to first 10 entries
       record.recent_timestamps = [NOW] + existing.recent_timestamps
           → keep only entries where timestamp >= (NOW - 24 hours)
   ELSE:
       record.schema_version = "1.0"
       record.signature   = the <normalized-error>|<category> signature string from Step A
       record.signature_hash = <signature-hash>
       record.title       = title from investigation
       record.category    = category from investigation
       record.count       = 1
       record.first_seen  = NOW
       record.last_seen   = NOW
       record.recent_run_urls = [CURRENT_RUN_URL]
       record.${{ github.aw.import-inputs.pattern_scope_field }} = (if SCOPE_REF: [SCOPE_REF] else [])
       record.recent_timestamps = [NOW]

   Additionally set every workflow-specific pattern field required by the
   workflow-specific instructions (if any), in both branches.
   ~~~

   **Step D — Write the file:**
   Write `record` as JSON to `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns/<signature-hash>.json`. Overwrite the file completely with the new content.

   **Step E — MANDATORY verification (read-back check):**
   Immediately after writing, read the file back and verify:
   - the record validates against `pattern.schema.json` (run the validation procedure from the "Artefact schemas" block)
   - `schema_version` equals `"1.0"` and `signature_hash` matches the file name
   - `count` equals the value you just computed (NOT 1 unless this is genuinely the first occurrence)
   - `recent_timestamps` contains the current timestamp `NOW` as the first entry
   - `last_seen` equals `NOW`
   - `first_seen` has NOT changed from `existing.first_seen` (if file existed before)
   - every additional check listed in the workflow-specific instructions passes

   If any check fails, you have a bug in your write logic. Fix it before proceeding.

   **Common failure modes to avoid:**
   - Writing `count: 1` because you forgot to read the existing file first
   - Writing `count: 1` because you recomputed the signature hash differently (different normalization) and created a new file instead of updating the old one
   - Omitting `recent_timestamps` entirely (this breaks recurrence detection)
   - Setting `first_seen` to NOW when the file already existed
   - Forgetting to include the current timestamp in `recent_timestamps`

3. **Build Statistics Snapshot**: After step 2, aggregate all `.json` files under `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/patterns/` into the statistics fields for `notify_teams`. For each file:
   - Read and parse the JSON
   - Use the `count`, `first_seen`, `last_seen`, `title`, `category` values AS-IS from the file (do NOT recompute them)
   - The current failure's pattern MUST report `count == notify_teams.occurrence_count`

   Sort patterns by `count` descending (ties broken by most recent `last_seen`). Format as the markdown table and JSON described in the Output Requirements section.

   **Validation before calling notify_teams:** Read back the current pattern file one more time. The `count` field in the file MUST equal the `occurrence_count` value you are about to pass to `notify_teams`. If they differ, go back to Step B and redo the update.

4. **Save Artifacts**: Store detailed logs and analysis in the cached directories.

5. **Retention & Compaction (MANDATORY — keeps the knowledge base under the repo-memory caps)**:

   The repo-memory branch enforces hard `max-file-count` (1000) and `max-file-size` (1 MiB) caps on every push; once either is exceeded, **the entire push is rejected** and this run's investigation — and all future ones — silently fail to persist. To stay under them, prune the two unbounded, low-value-over-time datasets down to rolling windows **after** writing this run's artefacts (steps 1–3). No permanent signal is lost: long-term recurrence lives in the per-signature `patterns/*.json` `count`/`first_seen`/`last_seen` fields, which are **never** pruned.

   Constants:
   - `INVESTIGATION_RETENTION_MAX = 300` — max per-run investigation record files to keep under `investigations/`.
   - `INDEX_RETENTION_MAX = 2000` — max entries to keep in `index.json` (≈600 KiB, comfortably under the 1 MiB file cap).

   **Step A — Prune raw investigation records:**
   - List every `*.json` file under `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/investigations/` **except** `index.json`.
   - If the count exceeds `INVESTIGATION_RETENTION_MAX`, delete the oldest files (order by the `YYYY-MM-DD-HH-MM-SS-sss` timestamp in the filename, oldest first) until exactly `INVESTIGATION_RETENTION_MAX` remain. **Never** delete the record you wrote in this run.
   - Deleting a raw record does not erase the failure from history: its lightweight reference stays in `index.json` (until the index prune below) and its recurrence is already folded into the pattern record.

   **Step B — Compact the index to its rolling window:**
   - Re-read `index.json` (the superset written in step 1a).
   - If `length(entries) > INDEX_RETENTION_MAX`, keep only the newest `INDEX_RETENTION_MAX` entries (sort by `timestamp` descending, take the first `INDEX_RETENTION_MAX`). This is the ONLY sanctioned place the index shrinks.
   - Set `total = length(entries)`, write the object back, then validate against `index.schema.json` and confirm this run's `investigation_id` is still present (it is one of the newest, so it must survive the trim).

   **Step C — Verify headroom (monitoring):**
   - Count all persisted files under `/tmp/gh-aw/repo-memory/default/${{ github.aw.import-inputs.slug }}/` and confirm the total is `< max-file-count` (1000).
   - Confirm `index.json` on disk is `< max-file-size` (1 MiB). If it is within ~10% of the cap despite the entry trim, lower the effective `INDEX_RETENTION_MAX` for this write and note it in the investigation.
   - Never leave the branch over any cap — an over-cap push is rejected wholesale and this run's investigation is lost. If pattern records alone approach `max-file-count`, flag it in the Teams report so a maintainer can raise the cap or archive old signatures.
