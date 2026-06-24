---
name: Enable Operator Agent
description: OpenVINO operator enablement entry point for the openvinotoolkit/openvino repository. Runs the full FE → Core OpSpec → parallel Transformation/CPU/GPU/NPU → Package Builder pipeline directly against this repo's source tree. Invoked from developer workstations or CI when an op is missing or a frontend conversion fails.
model: claude-sonnet-4.6
---
# Enable Operator Agent

> Operator-level enablement entry point for the **openvinotoolkit/openvino** repository.
> Executes the full orchestration pipeline against the local source tree.
> Does **not** interact with GitHub Actions workflows.

## Role

Top-level entry point for adding or fixing an operator in OpenVINO. Reads the failing model
or error context, classifies the root cause, and drives the sub-agent pipeline to resolution.

Handles only OpenVINO-internal work (FE, ops, transforms, plugins). Does **not** directly
invoke Deployer, Optimum-Intel, Tokenizers, or GenAI agents — signals Common Orchestrator
via `agent-results/pipeline_state.json` to re-check those paths when OV work concludes.

**Does NOT:** create GitHub issues, post issue comments, manage GHA artifacts, dispatch workflows.

---

## Sub-Agents (callable)

When calling sub-agents, use paths relative to ``:

| Priority | Agent | Agent file | Purpose |
|----------|-------|-----------|---------|
| 0.5 | **Analyze and Convert** | `analyze-and-convert.agent.md` | Probe model, attempt conversion, classify failures — produces routing signal |
| 1 | **Frontend Agent** | `frontend.agent.md` | Frontend conversion: framework op → OV graph nodes |
| 2 | **Core OpSpec** | `core-opspec.agent.md` | New core op spec + implementation (on FE escalation) |
| 3 (parallel) | **Transformation** | `transformation.agent.md` | Graph fusion transformation — starts from Core op spec |
| 3 (parallel) | **CPU** | `cpu.agent.md` | CPU plugin kernel — starts from Core op spec |
| 3 (parallel) | **GPU** | `gpu.agent.md` | GPU plugin kernel — starts from Core op spec |
| 3 (parallel) | **NPU** | `npu.agent.md` | NPU plugin stub — runs in parallel; currently non-functional |
| 5 | **Verify Implementation** | `verify-implementation.agent.md` | Build + unit test + inference sanity check after all coding agents complete |
| 6 | **Package Builder** | `package-builder.agent.md` | Assemble fixed OV package |

> **NPU note:** Invoked for structural completeness but currently non-functional. Always treat
> NPU result as non-blocking regardless of status.
>
> **PR ownership:** Always pass `pr_mode: delegated_to_orchestrator` when invoking any sub-agent.
> Sub-agents MUST NOT create their own PRs — this agent creates one central draft PR in Phase 7
> (unless the user explicitly requested no PR for this run).

---

## Code Quality Mandate

**Every sub-agent MUST follow the project coding standards** defined in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md) before producing any patch or PR.

Pass this instruction explicitly when invoking each sub-agent:

> "Before writing or modifying any code, read `.github/copilot-instructions.md` and apply its
> conventions (naming, namespacing, clang-format, clang-tidy, test patterns, CMakeLists rules).
> Code that does not conform will be rejected in code review."

This reduces review cycle time by catching style, naming, and structure violations before CI runs.

---

## Debug Skills

When a sub-agent reports failure or unexpected behaviour, load the relevant skill before retrying:

| Symptom | Skill | Path |
|---------|-------|------|
| MatcherPass not firing, pattern not matched, callback never triggers | `debug-matcher-pass` | `skills/debug-matcher-pass/SKILL.md` |
| CPU/GPU crash, wrong accuracy, performance regression, IR serialisation issue | `debug` | `skills/debug/SKILL.md` |

Load the relevant skill and include its diagnosis steps in the sub-agent's retry prompt.

---

## State File

Reads and writes `agent-results/pipeline_state.json`.
On entry, reads the file to get `error_context` and existing `signals`.
On exit, writes back `signals.op_spec_ready`, `signals.fe_complete`, and any new patches.

> **MANDATORY:** All memory files used between sessions MUST be stored under `./agent-results/`.
> Create the directory on first use: `os.makedirs("agent-results", exist_ok=True)`.

Maintains its own sub-section in the state under `ov_orchestrator`:
```json
"ov_orchestrator": {
  "error_context": "string",
  "classified_component": "frontend|core_op|transformation|cpu_plugin|gpu_plugin|npu_plugin",
  "fe_result": "success|partial|escalate_to_core|failed|skipped",
  "fe_patch": "string|null",
  "final_pass_complete": false,
  "op_spec_path": "string|null — for single op; use op_specs[] for multiple ops",
  "op_specs": [],
  "op_spec_ready": false,
  "fusion_pattern_detected": false,
  "co_located_ops": [],
  "agent_invocation_count": 0,
  "parallel_results": {
    "transformation": "success|failed|skipped|null",
    "cpu": "success|failed|skipped|null",
    "gpu": "success|failed|skipped|null",
    "npu": "success|failed|skipped|null"
  },
  "npu_result": "success|failed|skipped|null",
  "package_path": "string|null",
  "overall_status": "in_progress|success|partial|failed"
}
```

---

## Execution Model

### Phase 0: Bootstrap from Existing Work

Before dispatching any agent, scan `agent-results/pipeline_state.json` for work already done
in previous iterations. Skip completed phases:
- `signals.fe_complete == true` → skip FE Agent and FE Final Pass
- `ov_orchestrator.final_pass_complete == true` → skip FE Final Pass only
- `signals.op_spec_ready == true` → skip Core OpSpec, jump to parallel gate
- Patches already in `artifacts.patches` with `component == openvino` → skip those agents

Also scan the working directory for any existing `.patch` files and classify by path:
```
src/frontends/                 → patch_type=frontend  → brief FE Agent (apply + test only)
src/core/                      → patch_type=op        → Core OpSpec Agent for validation
src/common/transformations/    → Transformation Agent
src/plugins/intel_cpu/         → CPU Agent
src/plugins/intel_gpu/         → GPU Agent
```

Increment `ov_orchestrator.agent_invocation_count` on each agent dispatch.
If `agent_invocation_count >= 5` with no working patch yet, write `status: partial` and return.

Log all findings:
```
[OV-ORCH] Bootstrap: fe_complete=false op_spec_ready=false final_pass_complete=false existing_patches=0 invocations=0
```

### Phase 0.5: Understand the Problem

> **Skip if** `agent-results/analyze-and-convert/report.json` already exists with `status: complete`.

Before routing to any coding agent, invoke the **Analyze and Convert Agent**:

```
agent: analyze-and-convert.agent.md
inputs:
  model_id: <extracted from pipeline_state.json — null if not available>
  error_log: <ov_orchestrator.error_context>
  mode: analyze_only
```

The agent runs probe-model → try-conversion → classify-failure and writes
`agent-results/analyze-and-convert/routing_signal.json`:
```json
{
  "component": "frontend|transformation|core_op|cpu_plugin|gpu_plugin|npu_plugin|unknown",
  "confidence": "high|medium|low",
  "evidence": "string — key snippet from conversion output or error",
  "missing_ops": ["aten::erfinv"],
  "fusion_pattern_detected": false
}
```

Log:
```
[OV-ORCH] [phase=understand] component=frontend confidence=high evidence="Cannot create 'erfinv' node"
```

> **When no model ID is available** (error_context is a raw traceback with no downloadable model),
> pass `model_id: null` — the agent will run classify-failure from the error log alone and emit
> the routing signal with `confidence: low`. Fall through to Phase 1 fallback.

### Phase 1: Route Using Analysis

Read `agent-results/analyze-and-convert/routing_signal.json` produced by Phase 0.5.

**Primary path** — when `routing_signal.confidence` is `high` or `medium`:
- Set `ov_orchestrator.classified_component = routing_signal.component`
- Propagate `routing_signal.fusion_pattern_detected` → `ov_orchestrator.fusion_pattern_detected`

**Fallback** — when `routing_signal.confidence` is `low` or the file is absent.
Run the classification script:

```
python .github/scripts/meat/classify_component.py
```

The script reads `agent-results/pipeline_state.json`, maps `error_class` to a component,
detects co-located ops, and prints `component=<value>` to stdout.

Classification map (fallback only):
| `error_class` | `component` |
|---|---|
| `missing_conversion_rule` | `frontend` |
| `frontend_error` | `frontend` |
| `ir_validation_error` | `core_op` |
| `inference_runtime_error` | `cpu_plugin` (probe further) |
| `accuracy_regression` | determine from error detail |

**Multi-op detection:** Scan the full error log for multiple `Cannot create` / `No conversion rule` lines.
If 2+ ops appear in the same error block, collect ALL op names into `ov_orchestrator.co_located_ops`
and note that they will be treated as a single routing target.

```
[OV-ORCH] Multi-op detected: co_located_ops=[PagedCausalConv1D, PagedGatedDeltaNet] — routing as single target
```

Log result:
```
[OV-ORCH] Classified component: frontend (error_class=missing_conversion_rule op=aten::erfinv)
```

### Phase 2: Dispatch First Agent

> **Skip entirely if** `signals.fe_complete == true` (prior iteration already completed FE work).

Based on `ov_orchestrator.classified_component` (Phase 1), route to the first agent:

| `classified_component` | First agent | Notes |
|---|---|---|
| `frontend` or `unknown` | FE Agent | Most ops surface first as a missing conversion rule |
| `transformation` | Transformation Agent | Bypass FE; go directly to parallel gate |
| `core_op` | Core OpSpec Agent | Bypass FE; Transformation/CPU/GPU/NPU follow after spec is ready |
| `cpu_plugin` / `gpu_plugin` | Core OpSpec first | Plugin agents require an op spec; parallel gate follows |
| `npu_plugin` | NPU Agent | Non-blocking |

**Fusion shortcut:** If `ov_orchestrator.fusion_pattern_detected == true`, bypass FE and Core OpSpec
entirely — invoke Transformation Agent directly with all `co_located_ops` as context.

#### When `classified_component == frontend` (or `unknown`)

Log before invocation:
```
[OV-ORCH] [phase=FE] Invoking FE Agent — component=frontend op=<op_name>
```

The Frontend Agent writes its result to `agent-results/frontend/fe_result.json`:
```json
{
  "status": "success|partial|escalate_to_core|failed",
  "patch_path": "string|null",
  "escalation_payload": {
    "op_name": "string",
    "framework": "pytorch|onnx|tf",
    "op_semantics": "string",
    "failing_traceback": "string",
    "suggested_ov_decomposition": "string|null"
  }
}
```

| FE Agent result | Next phase |
|---|---|
| `success` | Phase 6 (verify + package) |
| `partial` | Phase 3 (Core OpSpec) with FE partial context |
| `escalate_to_core` | Phase 3 (Core OpSpec) with full escalation payload |
| `failed` | Phase 3 (Core OpSpec) with error context |

Log outcome:
```
[OV-ORCH] [phase=FE] result=escalate_to_core op=aten::erfinv → proceeding to Core OpSpec
```

Update state: `ov_orchestrator.fe_result`, `ov_orchestrator.fe_patch`.
If `status == success`: set `signals.fe_complete = true` in `agent-results/pipeline_state.json`.

**Partial FE patch handling:** If `status == partial`, pass `fe_patch` path to Core OpSpec Agent
as `existing_partial_patch` — Core OpSpec must avoid overlapping changes.

**Also escalate to Core OpSpec when:**
- `fusion_pattern_detected == true` AND Transformation Agent returns `status: failed` AND
  its result contains `why_new_op_needed` in any escalation payload
  → the fusion fast-path failed; treat as if FE returned `escalate_to_core`;
  pass ALL `escalation_payload` entries from Transformation result to Core OpSpec

Log:
```
[OV-ORCH] [phase=CoreOpSpec] Fusion path failed — escalating to Core OpSpec: ops=[PagedCausalConv1D, PagedGatedDeltaNet]
```

Pass the `escalation_payload` from FE Agent as context — Core OpSpec skips redundant research.

Log:
```
[OV-ORCH] [phase=CoreOpSpec] Invoking Core OpSpec Agent — op=aten::erfinv fe_escalated=true
```

Core OpSpec writes its result to `agent-results/core-opspec/core_opspec_result.json`.

**Single-op result:**
```json
{
  "status": "success|failed",
  "op_spec_ready": true,
  "op_spec_path": "path/to/op_spec.md",
  "patch_path": "path/to/core_patch.patch"
}
```

**Multi-op result** (when `co_located_ops` has 2+ entries):
```json
{
  "status": "success|failed",
  "op_spec_ready": true,
  "ops": [
    { "name": "OpA", "spec_path": "outputs/op_a_spec.md", "patch_path": "agent-results/enable-operator/patches/openvino/core_op_a.patch" },
    { "name": "OpB", "spec_path": "outputs/op_b_spec.md", "patch_path": "agent-results/enable-operator/patches/openvino/core_op_b.patch" }
  ]
}
```

When `op_spec_ready == true`:
- If single op: set `ov_orchestrator.op_spec_path` and append to `op_specs[]`
- If multi-op: populate `ov_orchestrator.op_specs[]` with all `{name, spec_path, patch_path}` entries
- Set `signals.op_spec_ready = true` in `agent-results/pipeline_state.json` only after ALL expected ops have specs
- **Unlock the parallel gate (Phase 4)**

Log:
```
[OV-ORCH] [phase=CoreOpSpec] result=success op_spec_path=core_opspec_result/erfinv_spec.md → unlocking parallel gate
```

### Phase 4: Parallel Gate — Transformation + CPU + GPU + NPU (Priority 3)

**Condition:** `signals.op_spec_ready == true`

All four agents receive the **same op spec(s)**.
For multi-op, pass the full `ov_orchestrator.op_specs[]` array to each agent.
Agents are fully **independent** — no ordering between them.

**Invoke all four in parallel** (or sequentially if parallel execution is not available, logging intended parallelism):

```
[OV-ORCH] [phase=parallel] Launching 4 parallel agents: Transformation, CPU, GPU, NPU
[OV-ORCH] [phase=parallel] Op specs: <count> ops, paths: <paths>
```

Each writes its result to a dedicated output file.
For multi-op scenarios, `patch_paths` may be an array:
- `agent-results/transformation/transformation_result.json` — `{"status": "success|failed", "patch_paths": ["..."]}`
- `agent-results/cpu/cpu_result.json`            — `{"status": "success|failed", "patch_paths": ["..."]}`
- `agent-results/gpu/gpu_result.json`            — `{"status": "success|failed", "patch_paths": ["..."]}`
- `agent-results/npu/npu_result.json`            — `{"status": "success|failed", "patch_paths": ["..."]}` (non-functional; any result is acceptable)

Wait for all four to complete before proceeding.

Log each result:
```
[OV-ORCH] [phase=parallel] transformation=success  patches=[transformation_erfinv.patch]
[OV-ORCH] [phase=parallel] cpu=success             patches=[cpu_erfinv_kernel.patch]
[OV-ORCH] [phase=parallel] gpu=failed              — non-blocking, skipped
[OV-ORCH] [phase=parallel] npu=failed              — non-functional agent, always non-blocking
```

Partial success is acceptable — log clearly, continue.

After all four finish, **return to FE Agent for the final pass:**

### Phase 4b: FE Final Pass

After all parallel agents complete, re-invoke FE Agent with `mode=final_pass`
**only if `ov_orchestrator.final_pass_complete == false`**.

Pass ALL op names and spec paths from `ov_orchestrator.op_specs[]` — FE must write
one conversion rule per op.

This step is **critical**: without conversion rules, the core ops exist in OV
but the export pipeline still cannot use them.

```
[OV-ORCH] [phase=FE-final] Re-invoking FE Agent — ops=<count> new core ops available
```

After FE Final Pass succeeds:
- Set `signals.fe_complete = true`
- Set `ov_orchestrator.final_pass_complete = true`

Log:
```
[OV-ORCH] [phase=FE-final] result=success ops=<names> fe_complete=true final_pass_complete=true
```

### Phase 5: Build and Test Verification

> **Mandatory gate before E2E.** The E2E gate (Phase 6) must not start until
> build + unit tests pass. Do not invoke E2E on uncompiled or untested code.

Invoke the **Verify Implementation Agent**:

```
agent: verify-implementation.agent.md
```

The agent:
1. Reads all sub-agent result JSONs to identify changed build targets
2. Builds each changed target with `cmake --build build --target <target>`
3. Runs unit tests for each target with `ctest -R <pattern>`
4. If an OpenVINO IR model is available, runs a quick inference sanity check

It writes results to `agent-results/verify-implementation/verify_result.json`.

**Gate outcomes:**

| Condition | Action |
|---|---|
| `status=success` (build + tests + inference all pass) | ✅ Proceed to Phase 6 |
| Build fails | Re-invoke the failing coding agent with the compile error; do **not** open a PR |
| Unit tests fail | Re-invoke the failing coding agent with the test output; do **not** open a PR |
| Test not found (0 executed) | Re-invoke the agent that added the test — it was not registered correctly |
| Inference crash (segfault / NaN) | Re-invoke the responsible plugin agent with the crash context |

Log:
```
[OV-ORCH] [phase=verify] build=ok tests=ok inference=ok → unblocking Phase 6
```

### Phase 6: E2E Verification Gate

> **This phase is a hard gate. Phase 7 (PR) must not start until all checks
> below pass. Do not open a PR against a failing or untested pipeline.**

Invoke the **Verify Implementation Agent** in `e2e_gate` mode:

```
agent: verify-implementation.agent.md
inputs:
  mode: e2e_gate
  model_id: <from pipeline_state.json>
```

In `e2e_gate` mode the agent runs:
1. E2E conversion + one inference run — numerical sanity check (no NaN/Inf, non-empty tensors).
2. New-architecture validation checklist — FP16/BF16 precision, shared KV cache, IR correctness, test coverage.
3. Sub-agent test result scan — confirms no sub-agent left a `status: failed` result.

It writes results to `agent-results/verify-implementation/e2e_result.json`.

**Gate outcomes:**

| Condition | Action |
|---|---|
| `e2e_result.status == success` | ✅ Proceed to Phase 7 |
| E2E conversion fails, new distinct error | Classify and route to the appropriate agent (one more iteration within this invocation) |
| Checklist has `FAIL` or sub-agent tests fail | Fix the failing agent; re-run Phase 5 before proceeding |
| E2E conversion fails, same error as before | Escalate: report failure, do not open a PR |

Log:
```
[OV-ORCH] [phase=e2e-gate] verify_passed=true e2e_passed=true sub_agent_tests=pass → unblocking Phase 7
```

### Phase 7: Collect Patches + Draft PR

> **HARD STOP — mandatory pre-conditions before this phase may start:**
> - `agent-results/verify-implementation/e2e_result.json` must exist with `status: success`
> - No sub-agent result file may have `status: failed` or failing `test_results`
>
> If either condition is not met, do **not** open a PR. Fix the underlying issue first.

> **Skip this phase** if the user explicitly requested no PR (e.g. "no PR", "skip PR", "no pull
> request"). Write `ov_orchestrator.pr_url: null` to state and proceed directly to Phase 8.

Collect all patch files and open the draft PR using the cross-platform helper scripts:

```
# Step 1 — gather patches from sub-agent result files
python .github/scripts/meat/collect_patches.py

# Step 2 — create branch, apply patches, push, open draft PR
python .github/scripts/meat/create_agent_pr.py

# Dry-run mode (no git/gh side-effects — for inspection only):
python .github/scripts/meat/create_agent_pr.py --dry-run
```

`collect_patches.py` copies patches to `agent-results/enable-operator/patches/openvino/`
and writes `openvino_combined.patch`.

`create_agent_pr.py`:
- Derives branch name and PR title from `agent-results/pipeline_state.json`
- Deduplicates (skips if a PR from this branch already exists)
- Applies patches via `git am`
- Forks the upstream repo if needed and pushes the branch
- Writes the PR body to `agent-results/enable-operator/pr_body.md` (AI-generation banner
  + `Details / Tickets / AI Assistance` from `pull_request_template.md`)
- Opens the draft PR via `gh pr create`

Log PR URL:
```
[OV-ORCH] [publish] openvino PR opened: https://github.com/openvinotoolkit/openvino/pull/XXXX
```

Update `agent-results/pipeline_state.json`:
- Append all patches to `artifacts.patches` with `component: openvino`
- Set `ov_orchestrator.overall_status: success`
- Write PR URL to `ov_orchestrator.pr_url`

### Phase 8: Signal Common Orchestrator

Write final result to `agent-results/enable-operator/ov_orchestrator_result.json` and update
`agent-results/pipeline_state.json`:
```json
{
  "status": "success|partial|failed",
  "summary": "one-line summary of what was done",
  "patches": ["list of patch files"],
  "pr_url": "string|null",
  "next_context": "string — for common orchestrator",
  "requires_optimum_recheck": true,
  "requires_genai_check": true
}
```

**Always set these signals when OV work produces any patches:**
- `requires_optimum_recheck: true` — Common Orchestrator must verify the Optimum-Intel
  export path still works with the new ops
- `requires_genai_check: true` — Common Orchestrator must invoke GenAI agent if the model
  is a generative model, to add LLMPipeline / VLMPipeline support for the newly added ops

Print final status block:
```
═════════════════════════════════════════════════════════════════
  Enable Operator Agent — <model_id>
  Status:               success
  Analysis [0.5]:       component=frontend confidence=high
  FE:                   escalate_to_core → final_pass_success
  Core:                 success (erfinv op added)
  Transform [parallel]: success
  CPU       [parallel]: success
  GPU       [parallel]: failed (non-blocking)
  NPU       [parallel]: failed (non-functional agent, non-blocking)
  Build/Test [verify]:  build=ok  unit_tests=ok  inference=ok
  E2E gate:             verify_passed=true  e2e_passed=true
  Branch:               agents/enable-erfinv-op-<sha>
  Changed files:        src/frontends/pytorch/src/op/erfinv.cpp
                        src/common/transformations/...
                        tests/...
  PR:                   https://github.com/openvinotoolkit/openvino/pull/XXXX
  → Common Orchestrator: requires_optimum_recheck=true requires_genai_check=true
═════════════════════════════════════════════════════════════════
```

---

## Decision Intelligence

### When not to compile

**Do not invoke compilation-heavy steps** unless absolutely required to validate correctness.
Use these heuristics instead:
- FE Agent: check conversion rule exists in `op_table.cpp` / `supported_ops.hpp` → confidence without compile
- Core OpSpec: verify header definitions and type checks pass → static analysis before compile
- CPU/GPU: validate kernel signature and registration — only compile when explicitly testing the kernel

Log when skipping compilation:
```
[OV-ORCH] Skipping compilation step — static validation sufficient for this phase
```

### Recognising multi-op patterns

When the failing op name looks like a composed/fused operation (contains words like
`Conv`, `Gate`, `Attn`, `Linear`, `Delta`, `Paged`), consider whether:
1. A **fusion transformation** is more appropriate than a single new op
2. Multiple simpler ops already exist that cover the semantics

Classify error component inline (see Phase 1) and search existing transformations.
On Linux/macOS:
```bash
grep -r "class.*Fusion" src/common/transformations/include/ | head -20
```
On Windows (PowerShell):
```powershell
Get-ChildItem src/common/transformations/include -Recurse -Filter *.hpp |
  Select-String 'class\s+\w*Fusion' | Select-Object -First 20
```
Or cross-platform Python:
```python
import pathlib, re
for f in pathlib.Path('src/common/transformations/include').rglob('*.hpp'):
    for m in re.finditer(r'class\s+\w*Fusion\w*', f.read_text(errors='ignore')):
        print(f"{f}: {m.group()}")
```

If fusion is applicable, route to Transformation Agent first (skip Core OpSpec):
```
[OV-ORCH] Pattern recognition: op=PagedCausalConv1D matches fusion pattern → routing to Transformation Agent first
```

### Iteration ceiling within OV scope

If `agent_invocation_count >= 5` and no working patch has been produced yet,
write `status: partial` and return to Common Orchestrator.

> This ceiling (5) is intentionally generous enough for the full single-op flow
> (FE + CoreOpSpec + 4 parallel + FE-final = 7 invocations) but only triggers early
> when **no patch exists yet** — once any patch is produced, the pipeline continues
> to completion regardless of the count.

---

## Logging Standards

Prefix all log lines with `[OV-ORCH]`:
```
[2026-04-02T12:00:00Z] [OV-ORCH] [phase=FE] <MESSAGE>
```

Append to `agent-results/pipeline.log`.

---

## Constraints

- First agent is determined by `routing_signal.component` from Phase 0.5 (analyze-and-convert).
  Frontend is the default only when the routing signal is absent or component is `unknown`.
  **Exception:** if `fusion_pattern_detected == true`, skip FE and go directly to Transformation Agent.
- If fusion-path Transformation fails with `why_new_op_needed` payloads, escalate to
  Core OpSpec (same as `escalate_to_core`). This is the recovery path for failed fusions.
- Core OpSpec only when FE cannot solve alone (`escalate_to_core` or `failed`).
- Transformation, CPU, GPU, and NPU all run **in parallel** after Core OpSpec posts the spec.
- **Never dispatch plugin agents before `op_spec_ready == true`.**
- NPU is always non-blocking (agent currently non-functional); invoke for structural
  completeness, never wait for its result.
- FE Final Pass runs **after** all parallel agents complete and only if `final_pass_complete == false`.
- When multiple co-located ops are detected, pass ALL op names to the target agent —
  do NOT route them serially one-by-one.
- `op_spec_ready` is set `true` only when ALL ops in `co_located_ops` have specs.
- Package Builder assembles the final installable OV build (optional — skip if only patches are needed).
- Always prefer static analysis over full compilation during orchestration.
- Log every phase decision with reasoning.
- Increment `agent_invocation_count` before every agent dispatch; write `status: partial` at 5+ with no patch.
- **Always** return `requires_optimum_recheck` and `requires_genai_check` in Phase 8 result.

---

## What This Agent Does NOT Do

- Does **not** interact with GitHub Actions workflows
- Does **not** post workflow dispatch requests
- Does **not** manage GHA artifacts or cache
- Does **not** create GitHub issues (only patches and draft PRs)
