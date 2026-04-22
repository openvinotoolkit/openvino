# Agent scripts — local runner

Scripts for running the OpenVINO coding agents from `.github/agents/` on your
local machine using the [GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/cli-getting-started).

---

## Quick Start — Local

### 1. Install prerequisites

Run from the repo root:

```bash
python .github/scripts/meat/setup.py
```

This checks and installs:
- Python 3.10+
- `gh` (GitHub CLI)
- `copilot` CLI
- Copilot authentication (opens browser login if needed)
- Python dependencies from `requirements.txt` (if present)

### 2. Write a context file

Create a plain text (or Markdown) file describing the problem.
One operator or one model per file.

**Operator enablement:**

```markdown
Operator: aten::erfinv
Model: Qwen/Qwen3-0.6B
Error: No conversion rule for aten::erfinv

erfinv computes the element-wise inverse error function.
PyTorch docs: https://pytorch.org/docs/stable/generated/torch.erfinv.html

The op appears in the attention normalization path.
Single tensor in, single tensor out. dtype: float32/float16.
```

Both `Operator:` and `Model:` are important — the agents need a model to
reproduce and test the fix.

### 3. Run

**Full operator enablement pipeline** (FE → Core OpSpec → Transformation/CPU/GPU/NPU):

```bash
python .github/scripts/meat/enable_operator.py my_op.md
```

**Run a single specialist agent:**

```bash
# Triage — export + classify failure
python .github/scripts/meat/run_deployer.py my_context.md
python .github/scripts/meat/run_analyze_and_convert.py my_context.md

# OpenVINO core pipeline
python .github/scripts/meat/run_pytorch_fe.py my_context.md
python .github/scripts/meat/run_core_opspec.py my_context.md
python .github/scripts/meat/run_transformation.py my_context.md
python .github/scripts/meat/run_cpu.py my_context.md
python .github/scripts/meat/run_gpu.py my_context.md
```

All output goes to `agent-results/<agent-name>/` in the working directory:
- `session.md` — full agent session transcript
- `patches/` — generated `.patch` files ready to apply
- `pipeline_state.json` — shared state read/written by all agents

### 4. Apply patches

```bash
# Check first
git apply --check agent-results/enable-operator/patches/openvino/*.patch

# Apply
git apply agent-results/enable-operator/patches/openvino/*.patch

# If whitespace issues
git apply --whitespace=fix agent-results/enable-operator/patches/openvino/*.patch
```

---

## Agent pipeline map

```
enable_operator.py
    │
    ├─ run_pytorch_fe.py       ← missing_conversion_rule / frontend_error
    ├─ run_core_opspec.py      ← FE escalates (no existing OV op)
    │   └─ (parallel)
    │       ├─ run_transformation.py
    │       ├─ run_cpu.py
    │       └─ run_gpu.py
    │
run_deployer.py                ← first-attempt export + classify
run_analyze_and_convert.py     ← deep probe + strategy matrix + routing signals
```

---

## Tips & Tricks — Writing Effective Context

The quality of the context file directly affects how well the agents perform.

### Do

- **Include the exact error message or traceback.** Agents parse these to make
  routing decisions. A partial traceback is better than none.
- **Name the operator explicitly** (`aten::erfinv`, not "some math function").
- **Link to documentation** (PyTorch op docs, paper, HF model card).
- **Mention the model ID** — agents use it to reproduce and test the fix.
- **State what you already tried** — saves the agent from repeating failed approaches.

### Don't

- Don't paste entire 10 000-line logs. Trim to the relevant section.
- Don't ask for multiple unrelated things in one context file. One operator
  or one model per run.
- Don't include instructions about how agents should work — they already have
  their own instructions. Focus on the *problem*, not the *process*.

### Example: good context for a frontend issue

```markdown
Model: Qwen/Qwen3-0.6B
Operator: aten::erfinv
Task: text-generation

Export fails with:
  RuntimeError: No conversion rule found for aten::erfinv
  at openvino/frontend/pytorch/ts_decoder.py:287

erfinv computes the element-wise inverse error function.
PyTorch docs: https://pytorch.org/docs/stable/generated/torch.erfinv.html

The op appears in the attention normalization path. Single tensor in, single
tensor out. dtype: float32/float16.
```

### Example: good context for a Core OpSpec escalation

```markdown
Model: Qwen/Qwen3-0.6B
Operator: aten::erfinv

FE agent escalated — no existing OV op covers erfinv semantics.
Escalation payload:
  op_name: aten::erfinv
  op_semantics: element-wise inverse error function, float tensor in/out
  suggested_ov_decomposition: null
```

---

## Authentication

The `copilot` CLI requires a GitHub account with an active Copilot subscription.

`setup.py` handles login automatically. To authenticate manually:

```bash
copilot login
```

Or set an environment variable with a Personal Access Token (scopes: `read:user`, `copilot`):

```bash
# Linux / macOS
export COPILOT_GITHUB_TOKEN=ghp_<your-token>

# Windows PowerShell
$env:COPILOT_GITHUB_TOKEN = 'ghp_<your-token>'
```

---

## See also

- Agent definitions: [`.github/agents/`](../agents/)
- Copilot CLI reference: https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
