# OpenVINO Orchestrator Agent

## Role

Second-level orchestrator for **core OpenVINO** issues.
Called by the Common Orchestrator when the error is classified as a missing op,
frontend conversion failure, transformation issue, or plugin problem.

## Called by

- **Common Orchestrator** (Step 5)

## Agents (callable)

These are the **only** agents this orchestrator may invoke.
Order matters - issues are resolved sequentially, re-testing after each fix.

| Priority | Agent | Workflow | Purpose |
|----------|-------|----------|---------|
| 1 | **PyTorch FE** | `pytorch-fe.yml` | Frontend conversion: torch → OV graph |
| 2 | **Core OpSpec** | `core-opspec.yml` | Missing op specification + implementation |
| 3 | **Transformation** | `transformation-agent.yml` | Graph-level optimisation passes |
| 4 | **CPU** | `cpu-agent.yml` | CPU plugin validation + fixes |
| 5 | **GPU** | `gpu-agent.yml` | GPU plugin validation + fixes |
| 6 | **NPU** | `npu-agent.yml` | NPU plugin validation + fixes |
| 7 | **Package Builder** | `package-builder.yml` | Assemble fixed OV package |

> **Boundary:** This orchestrator does NOT call Deployer, Optimum-Intel,
> Tokenizers, or GenAI agents. Those are managed exclusively by the
> **Common Orchestrator**.

## Execution Model

### Step 1: Identify failing component

Parse the error context received from Common Orchestrator.
Classify into: `frontend`, `core_op`, `transformation`, `cpu_plugin`,
`gpu_plugin`, `npu_plugin`.

### Step 2: Fix sequentially

For each identified issue (in priority order):
1. Call the corresponding agent.
2. If the agent returns success - **re-test** the full pipeline to check
   for cascading issues.
3. If new issues surface - classify and add to the fix queue.
4. Repeat until all issues are resolved or all agents are exhausted.

### Step 3: Build package

Once all fixes succeed:
1. Call **Package Builder** to assemble the fixed OpenVINO package.
2. Return the package spec (branches, build instructions) to the
   Common Orchestrator.

### Step 4: Report

Return to Common Orchestrator:
- `status`: `success` or `failed`
- `package`: package spec (if success)
- `remaining_issues`: list of unresolved issues (if failed)
- `fixes_applied`: list of `{component, branch, description}`

## Artifact Manifest (Tier-2)

The OV Orchestrator and all its sub-agents share the same run-scoped manifest
(`meat-manifest-<run_id>`) as Tier-1. No separate manifest is needed.

### Bootstrap at entry

At the start of any Tier-2 job, download and bootstrap from the manifest:
```bash
python scripts/collect_artifacts.py bootstrap --manifest meat_manifest.json | bash
```
This installs patched `optimum-intel`, `transformers` overrides, and any earlier
Tier-2 patches automatically — each sub-agent builds on the previous one's work
without explicit wiring.

### Reuse existing IR when available

If the Deployer (in Pass 1 or 2) recorded a `model_ir` entry in the manifest,
download that IR instead of re-exporting:
```bash
IR_URL=$(python scripts/collect_artifacts.py get \
  --manifest meat_manifest.json --type model_ir --field artifact_url)
if [ -n "$IR_URL" ]; then
  # download OV IR — skip optimum-cli re-export, saves several minutes
else
  optimum-cli export openvino ...  # full export with bootstrapped packages
fi
```

### Every sub-agent records its output

```bash
python scripts/collect_artifacts.py add \
  --agent pytorch-fe --pass 1 \
  --type patch --component openvino \
  --artifact-name "pytorch-fe-patch-${GITHUB_RUN_ID}" \
  --branch "fix/add-scatter-reduce-rule" \
  --install-cmd "pip install git+https://github.com/openvinotoolkit/openvino@fix/add-scatter-reduce-rule" \
  --description "Added aten::scatter_reduce conversion rule to PyTorch FE"
```

Because the manifest is re-uploaded after each sub-agent, later agents (CPU, GPU,
NPU) automatically inherit earlier patches via the `bootstrap` call. The Package
Builder (priority 7) reads all `patch`-type entries to know which branches to merge:
```bash
python scripts/collect_artifacts.py get \
  --manifest meat_manifest.json --type patch --field branch
```

## Constraints

- Agents communicate only through this orchestrator - no direct cross-calls.
- Fix one component at a time; re-test before moving to the next.
- Always attempt the full priority list before declaring failure.
- Report all intermediate results for traceability.
- Every sub-agent that produces a patch or wheel must update the manifest before returning.
