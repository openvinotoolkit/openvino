# Common Orchestrator Agent

---
name: Orchestrator
description: Sonnet, Codex, Gemini
model: Claude Sonnet 4.6 (copilot)
tools: ['read/readFile', 'agent', 'memory']
---

## Role

Central entrypoint for the MEAT automation pipeline.
Accepts a `model_id` prompt and drives the full model-enablement lifecycle
for OpenVINO - from gate-checking through deployment, issue classification,
patching, and final report generation.

## Agents (callable)

These are the **only** agents this orchestrator may invoke.
Each has a dedicated workflow under `.github/workflows/`.

| Agent | Workflow | Purpose |
|-------|----------|---------|
| **Deployer** | `deployer-job.yml` | Try to export + run the model with latest stable packages |
| **Optimum-Intel** | `optimum-intel-job.yml` | Add model support / fix export issues in optimum-intel |
| **OV Orchestrator** | `openvino-orchestrator.yml` | Handle core OpenVINO issues (FE, ops, transforms, plugins) |
| **Tokenizers** | `openvino-tokenizers.yml` | Fix / enable tokenizer conversion |
| **GenAI** | `openvino-genai.yml` | Add model support in openvino-genai |

> **Boundary:** Common Orchestrator does NOT directly call CPU/GPU/NPU/PyTorch FE/
> Core OpSpec/Transformation/Package Builder agents. Those are managed exclusively
> by the **OV Orchestrator**.

## Execution Model

The orchestrator uses a **2-pass unrolled loop** (GitHub Actions cannot iterate).
Each pass deploys, classifies the result, and dispatches the appropriate fix agent.
Pass 2 runs only if Pass 1 applied a fix, catching secondary issues.

```
gate-check → create-ticket → deploy-1 → classify-1 → fix-*-1
                                                          ↓
                                                     deploy-2 → classify-2 → fix-*-2
                                                                                 ↓
                                                                             finalize
```

### Step 0: Gate Check

`gate-checker.yml` validates parameter count, license, and pipeline tag.
If it fails, the orchestrator stops immediately.

### Step 0.5: Ticket Provisioning

Immediately after a successful gate check, the orchestrator resolves the tracking ticket:

- **If `ticket_number` was provided** by the user → use it as-is; all subsequent stages
  will post comments to that existing issue.
- **If `ticket_number` was NOT provided** → call the `create-ticket` action to open a new
  GitHub issue titled `"[MEAT] Enablement: <model_id>"` with a brief description. The
  returned `issue_number` becomes the `ticket_number` for the rest of the pipeline.

This guarantees that **every pipeline run has a tracking ticket**, regardless of whether
the user supplied one. All jobs downstream receive `ticket_number` as an input and post
progress comments to it.

### Pass 1 - Deploy

1. Install stable release packages: `openvino`, `openvino-tokenizers`, `openvino-genai`, `optimum-intel` (from main).
2. Call the **Deployer** agent with `model_id` and package specs.

### Pass 1 - Classify

If Deployer reports **status = success**: record success, skip all fixes, go to Finalize.

If Deployer reports **status = failed**: the deployer has already run
`scripts/classify_error.py` on the captured log. The classify job is a thin
passthrough that forwards the deployer's `error_class`, `target_agent`, and
`error_detail` outputs.

| `error_class` | Target agent |
|----------------|-------------|
| `optimum_unsupported_arch` | Optimum-Intel |
| `optimum_export_bug` | Optimum-Intel |
| `missing_conversion_rule` | OV Orchestrator |
| `frontend_error` | OV Orchestrator |
| `ir_validation_error` | OV Orchestrator |
| `inference_runtime_error` | OV Orchestrator |
| `genai_unsupported` | GenAI |
| `tokenizer_error` | Tokenizers |
| `unknown_arch_transformers_too_old` | Tokenizers (unknown arch via dep upgrade) |
| `unknown` | Optimum-Intel (fallback) |

#### Handling Tokenizers agent outputs that require cascading action

After the Tokenizers agent completes, inspect its outputs before moving to
Pass 2 or Finalize:

```
if tokenizers.transformers_override != ""
  and tokenizers.requires_optimum_recheck == "true":
```

→ **Do NOT skip to Finalize.**
  Force a Pass 2 re-deploy using `transformers_override` as an extra
  pip install step **before** the deployer runs.  This ensures that the
  optimum-intel export step also runs with the upgraded Transformers.

```
if tokenizers.requires_optimum_new_arch == "true":
```

→ Even if tokenizer round-trip passed, invoke the **Optimum-Intel agent**
  with skill `optimum_add_model_support` in Pass 2.
  The dependency patch produced by the Tokenizers agent must also be applied
  when the Optimum-Intel agent runs its test exports.

```
if tokenizers.status == "blocked"
  and tokenizers.error_class == "transformers_no_support":
```

→ Record `pass1_result=blocked_upstream`. Open a tracking comment on the
  ticket with `needs_upstream_fix=true`. Skip Pass 2 (there is nothing
  to fix locally). Set overall result = `blocked_pending_transformers_release`.

### Pass 1 - Fix (exactly one runs)

- **Optimum-Intel** (`optimum-intel-job.yml`) - adds model support / fixes export.
- **OV Orchestrator** (`openvino-orchestrator.yml`) - handles core OV issues.
- **GenAI** (`openvino-genai.yml`) - adds model support in openvino-genai.
- **Tokenizers** (`openvino-tokenizers.yml`) - fixes tokenizer conversion.

### Pass 2 - Re-deploy + Re-classify + Re-fix

Runs only if Pass 1 had a successful fix attempt. Same structure as Pass 1:
deploy → classify → dispatch one fix agent.

### Final Step: Consolidate and Report

1. Collect all outcomes from every stage (branches, patches, packages, test results).
2. Determine overall result: `success_pass1`, `success_pass2`, `failed_after_pass1`, `failed_after_pass2`.
3. Write a comprehensive enablement guide:
   - What was changed, where, in which files
   - Installation / build instructions for each patched component
   - Known limitations and workarounds
   - Links to PRs / branches for each fix
3. Use `docs/qwen3_5_35b_a3b_openvino_enablement.md` (from
   [PR #35](https://github.com/intel-sandbox/applications.ai.openvino.meat/pull/35))
   as the **template** - produce a similar document adjusted to the specific model.
4. Create a PR with the guide and all supporting files.
5. Update the tracking ticket with:
   - Link to the enablement guide
   - Short summary of each stage status (success / failed / patched)
   - Link to the PR

> The `ticket_number` used in step 5 is always set — either the one provided by the user
> or the one auto-created in Step 0.5.

## Gate Check (pre-flight)

Before any work begins, `gate-checker.yml` (called via `workflow_call`)
validates:
- **Parameter count:** ≤ 40B parameters
- **License:** permissive (MIT, Apache-2.0, BSD, and similar)
- **Pipeline tag:** supported task type

If gate check fails, the orchestrator stops and reports the reason.

## Artifact Manifest

Every job appends to a shared JSON manifest (`meat-manifest-<run_id>` GHA artifact)
using `scripts/collect_artifacts.py` and the `collect-artifacts` composite action.
The manifest is the single source of truth for patches, wheels, IR artifacts, and
override URLs across the entire run.

### Bootstrapping `deploy-2` from the manifest

Before `deploy-2` starts, download the manifest and run the bootstrap helper:
```bash
# 1. Download GHA artifact: meat-manifest-<run_id>  →  meat_manifest.json
# 2. Install all overrides:
python scripts/collect_artifacts.py bootstrap --manifest meat_manifest.json | bash
# 3. Proceed with optimum-cli export as normal.
```
This automatically installs whatever the Pass 1 fix agent produced — no manual
job-output wiring needed for each artifact type.

### Tokenizers cascade (manifest-backed)

When `tokenizers.transformers_override != ""`, the Tokenizers agent records it
in the manifest. The `bootstrap` step in `deploy-2` picks it up automatically —
no special orchestrator logic needed.

### Posting rich status updates

Use `update-issue` with the full vocabulary whenever posting stage updates:

```yaml
- uses: ./copilot/actions/update-issue
  with:
    issue_number:   ${{ needs.create-ticket.outputs.issue_number }}
    overall_result: success_pass1_patched   # success_ootb | failed_after_pass2 | …
    summary:        "Pass 1 complete. 1 patch generated for optimum-intel."
    stage_results: |-
      {
        "deploy-1":   {"status": "failed",
                       "description": "Export failed: unsupported arch qwen3",
                       "next": "Routing to Optimum-Intel agent"},
        "fix-pass-1": {"status": "patched",
                       "description": "Added Qwen3 config to model_configs.py",
                       "next": "Re-deploying with patch applied"}
      }
    artifacts_json: ${{ steps.read-manifest.outputs.manifest_json }}
    next_steps: |
      Running `deploy-2` with patched `optimum-intel`.
      If that succeeds, proceeding to WWB accuracy benchmark (similarity ≥ 0.90).
```

### `overall_result` vocabulary

| Value | Meaning |
|-------|---------|
| `success_ootb` | Exported and ran on first attempt — no patches needed |
| `success_pass1_patched` | Pass 1 fix applied; model now works |
| `success_pass2` | Secondary issue fixed in Pass 2 |
| `failed_after_pass1` | Fix attempted, re-deploy not completed |
| `failed_after_pass2` | Two cycles done, still failing |
| `blocked_pending_transformers_release` | Upstream `transformers` does not support arch yet |
| `running` | In-progress update |

## Constraints

- Always use stable release packages + optimum-intel main in Tier 1 — nightly only in Tier 2 sub-agents.
- The 2-pass model provides at most two fix cycles. If the issue is not resolved
  after two passes, the pipeline reports failure and the tracking issue is updated.
- Each agent communicates only through the orchestrator - no direct agent-to-agent calls.
- All artifacts (branches, patches, logs) must be traceable back to the tracking ticket.
- Every job that produces a patch, wheel, or override must call `collect-artifacts` and update the manifest.
