---
name: submit-draft-pr
description: Optionally create a draft PR to the upstream repo from a local source directory. Used by coding agents when a local source path is available in the context file.
---

# Skill: Submit Draft PR

> **Optional.** Attempt this only when all of the following are true:
> - Your context file provides a local source path (e.g. `OpenVINO source code: /path/to/openvino`)
> - That path is accessible and is a git repository
> - `gh` CLI is available and authenticated (`gh auth status`)
>
> If any condition is not met — **skip silently** and log one line. Do not fail.

---

## When to invoke

After completing your main implementation work and writing results to
`agent-results/<agent>/`, call this skill to open a **draft PR** from your
changes to the upstream repo.

This is *in addition to* — never instead of — generating patch files and
writing results to `agent-results/`.

---

## Steps

### 1. Verify prerequisites

```bash
# Is gh available and authenticated?
gh auth status || { echo "[draft-pr] gh not available — skipping"; exit 0; }

# Is the source path a git repo?
[ -d "${SOURCE_PATH}/.git" ] || { echo "[draft-pr] ${SOURCE_PATH} is not a git repo — skipping"; exit 0; }
```

### 2. Choose a branch name

Use a descriptive kebab-case branch name derived from the work done:

```
fix/add-<op-name>-op              # core op or FE op
fix/add-<pass-name>-transformation
fix/<model-id>-export             # optimum-intel / tokenizers
fix/<component>-<short-desc>      # general
```

### 3. Run the helper script

```bash
python scripts/create_draft_pr.py \
  --repo-dir "${SOURCE_PATH}" \
  --branch   "<branch-name>" \
  --title    "<one-line description of the fix>" \
  --body-file agent-results/<agent>/agent_report.md \
  [--upstream <org/repo>]          # only if auto-detection might fail
```

The script will:
1. Detect whether `origin` is a fork or the upstream directly
2. Ensure a personal fork exists (creates one via `gh repo fork` if needed)
3. Create the feature branch (or reuse an existing one with the same name)
4. `git add -A && git commit` any staged/unstaged changes
5. Push to the fork (`git remote add fork <fork-url>`)
6. Open a **draft PR** to the upstream repo via `gh pr create --draft`
7. Print the PR URL on success

### 4. Log the result

After the script runs, record the outcome in `agent-results/<agent>/session.md`:

```
[draft-pr] PR opened: <pr_url>
```

or

```
[draft-pr] Skipped: <reason>
```

---

## Per-repo upstream table

| Agent | Upstream repo |
|---|---|
| transformation, core-opspec, pytorch-fe, cpu, gpu, npu | `openvinotoolkit/openvino` |
| optimum-intel | `huggingface/optimum-intel` |
| openvino-tokenizers | `openvinotoolkit/openvino_tokenizers` |
| openvino-genai | `openvinotoolkit/openvino.genai` |

Pass `--upstream <value>` only when auto-detection from the git remote is likely
to fail (e.g. the local clone uses an internal mirror URL).

---

## Important constraints

- Always create PRs as **drafts** — never ready-for-review.
- PR target is the **upstream** repo, not the fork.
- Do not push to `main`/`master` of any repo.
- If `git push --force-with-lease` fails (e.g. diverged history), skip and log — do not `--force`.
- Do not block the agent's completion on PR success — log the failure and continue.
