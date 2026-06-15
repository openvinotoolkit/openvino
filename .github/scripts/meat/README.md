# Agent scripts — local runner

Scripts for running the OpenVINO coding agents from `.github/agents-prototype/` on your
local machine using the [GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/cli-getting-started).

---

> [!WARNING]
> **Autonomous / unattended mode**
>
> These scripts invoke GitHub Copilot with `--no-ask-user` and `--autopilot`.
> The agent will **read, create, and modify files** in this repository
> **without asking for confirmation**.
>
> - Always run on a **clean branch** with no uncommitted work.
> - Review `agent-results/` after the run.
> - Apply generated patches with `git apply` and inspect them before committing.
> - **Do NOT blindly commit agent-generated changes.**

> [!TIP]
> **Consider running the agent inside an isolated sandbox.**
>
> Autopilot mode gives the agent unrestricted access to your filesystem, shell,
> and network. Sandboxing limits the blast radius of unexpected agent behaviour
> (runaway file writes, unintended network calls, accidental credential exposure).
>
> Possible options (not an exhaustive list):
> - **Qwen Code sandbox** — sandboxing feature of the Qwen Code CLI that restricts
>   filesystem and network access during agent runs (macOS Seatbelt or Docker/Podman):
>   https://qwenlm.github.io/qwen-code-docs/en/users/features/sandbox/
> - **gh-aw-firewall** — network firewall for agentic workflows that restricts
>   outbound HTTP/HTTPS to an allowlist of domains via a Squid proxy inside Docker:
>   https://github.com/github/gh-aw-firewall
> - **agent-sandbox** (Kubernetes SIGs) — Kubernetes CRD and controller for managing
>   isolated, stateful singleton workloads, ideal for AI agent runtimes:
>   https://github.com/kubernetes-sigs/agent-sandbox
> - **sandbox-runtime** (Anthropic) — lightweight OS-level sandboxing tool (no
>   container required) that enforces filesystem and network restrictions using native
>   OS primitives (macOS Seatbelt / Linux bubblewrap):
>   https://github.com/anthropic-experimental/sandbox-runtime

---

## Quick Start — Local

### 1. Install prerequisites

Run from the repo root:

```bash
# Linux — must be run as root to install system packages (gh, copilot CLI)
sudo python .github/scripts/meat/install_copilot_env.py

# macOS / Windows — no sudo needed
python .github/scripts/meat/install_copilot_env.py
```

This checks and installs:
- Python 3.10+
- `gh` (GitHub CLI)
- `copilot` CLI
- Copilot authentication (opens browser login if needed)

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
python .github/scripts/meat/run_agent.py deployer my_context.md
python .github/scripts/meat/run_agent.py analyze-and-convert my_context.md

# OpenVINO core pipeline
python .github/scripts/meat/run_agent.py frontend my_context.md
python .github/scripts/meat/run_agent.py core-opspec my_context.md
python .github/scripts/meat/run_agent.py transformation my_context.md
python .github/scripts/meat/run_agent.py cpu my_context.md
python .github/scripts/meat/run_agent.py gpu my_context.md

# List all available agents
python .github/scripts/meat/run_agent.py --list
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
    │  (delegates to run_agent.py enable-operator)
    │
    ├─ run_agent.py frontend         ← missing_conversion_rule / frontend_error
    ├─ run_agent.py core-opspec      ← FE escalates (no existing OV op)
    │   └─ (parallel)
    │       ├─ run_agent.py transformation
    │       ├─ run_agent.py cpu
    │       └─ run_agent.py gpu
    │
run_agent.py deployer              ← first-attempt export + classify
run_agent.py analyze-and-convert   ← deep probe + strategy matrix + routing signals
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

`install_copilot_env.py` handles login automatically. To authenticate manually:

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

## Using skills from VS Code Chat

Individual skills are also available as **slash commands** directly in the
VS Code Copilot Chat panel — no terminal or CLI required.

Each skill maps to a `.prompt.md` file in [`.github/prompts/`](../prompts/).
VS Code discovers these automatically and exposes them as `/skill-name`.

| Command | What it does |
|---|---|
| `/add-core-op` | Add a new op to the OpenVINO Core opset |
| `/add-fe-op` | Translate a framework op in a Frontend (ONNX/PyTorch/TF) |
| `/add-fusion-transformation` | Write a `MatcherPass`/`FunctionPass` fusion pass |
| `/add-cpu-op` | Implement a CPU plugin kernel (AVX2/AVX-512/AMX/oneDNN) |
| `/add-gpu-op` | Implement a GPU plugin OpenCL kernel |
| `/analyze-and-convert` | Probe a model and classify conversion failures |
| `/conversion-issues` | Diagnose and fix conversion errors |
| `/verify-conversion` | E2E gate — convert + run inference + sanity check |
| `/python-bootstrap` | Set up the Python environment (release or source build) |
| `/submit-draft-pr` | Create a draft PR with duplicate-PR guard |

**How to use:**

1. Open VS Code Copilot Chat (`Ctrl+Alt+I`).
2. Switch to **Agent** mode (dropdown next to the model selector).
3. Type `/` and select the skill from the autocomplete list.
4. Add your context after the command, e.g.:

```
/add-fe-op  Model: Qwen/Qwen3-0.6B  Operator: aten::erfinv
```

The skill instructions are loaded automatically — the agent follows the
step-by-step workflow defined in the corresponding skill file.

---

## See also

- Agent definitions: [`.github/agents-prototype/`](../agents-prototype/)
- Skill prompts: [`.github/prompts/`](../prompts/)
- Copilot CLI reference: https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
