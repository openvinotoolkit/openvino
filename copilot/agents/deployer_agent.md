# Deployer Agent

## Role

First-attempt deployment of a HuggingFace model to OpenVINO.
This agent installs stable release packages, exports the model via `optimum-cli`,
validates the IR, and runs a quick inference sanity check.

## Called by

- **Common Orchestrator** (Step 1)

## Inputs

| Input | Description |
|-------|-------------|
| `model_id` | HuggingFace model ID |
| `packages` | Package overrides (optional - defaults to stable release) |

## Execution

1. Install packages:
   - `openvino` (stable release)
   - `optimum-intel` (latest main: `pip install git+https://github.com/huggingface/optimum-intel.git`)
   - `openvino-genai` (stable release)
   - `openvino-tokenizers` (stable release)
2. Detect task via `scripts/detect_task.py` (`text-generation` -> `text-generation-with-past`).
3. Export:
   ```bash
   optimum-cli export openvino --model <model_id> --task <task> --weight-format fp16 ov_model
   ```
4. Validate IR files exist (`.xml`, `.bin`).
5. Run quick inference sanity check:
   ```python
   from optimum.intel import OVModelForCausalLM
   model = OVModelForCausalLM.from_pretrained("ov_model")
   # generate a few tokens
   ```
6. Optionally run WWB similarity check (threshold ≥ 0.9).

## Outputs

| Output | Description |
|--------|-------------|
| `status` | `success` or `failed` |
| `error_log` | Full error traceback (if failed) |
| `error_class` | 9-class category from `analyze_error.py` (LLM-based, falls back to regex): `optimum_unsupported_arch`, `optimum_export_bug`, `missing_conversion_rule`, `frontend_error`, `ir_validation_error`, `inference_runtime_error`, `genai_unsupported`, `tokenizer_error`, `unknown` |
| `target_agent` | Routing target derived from `error_class` |
| `error_detail` | One-line summary (from LLM analysis or regex match) |
| `analysis_report` | `error_analysis.md` — full Markdown report with LLM root-cause analysis, listing **all** identified errors with log excerpts; uploaded as artifact `error-analysis-<run_id>` and posted directly to the tracking issue |
| `ir_artifact` | Path to exported IR (if success) |

## Artifact Manifest

After each run the Deployer records its results in the run-scoped manifest
(`meat_manifest.json`, uploaded as GHA artifact `meat-manifest-<run_id>`).

**Before installing packages** — if the orchestrator supplies a manifest from a
previous pass, bootstrap from it first:
```bash
# Download meat-manifest-<run_id>, then:
python scripts/collect_artifacts.py bootstrap --manifest meat_manifest.json \
  | bash   # installs all patches/wheels from earlier agents
```

**On success** — add a `model_ir` entry:
```bash
python scripts/collect_artifacts.py add \
  --agent deployer --pass "$PASS_NUM" \
  --type model_ir --component openvino \
  --artifact-name "openvino-ir-${GITHUB_RUN_ID}" \
  --artifact-url  "$IR_ARTIFACT_URL" \
  --description   "OV IR for $MODEL_ID (export succeeded)"
```

**On failure** — add an `analysis` entry:
```bash
python scripts/collect_artifacts.py add \
  --agent deployer --pass "$PASS_NUM" \
  --type analysis --component openvino \
  --artifact-name "error-analysis-${GITHUB_RUN_ID}" \
  --description   "$ERROR_CLASS — $ERROR_DETAIL"
```

## Constraints

- Reports results back to the Common Orchestrator - does not classify or fix issues.
- Does not call other agents directly.
- Always uses stable release packages unless overridden by orchestrator.
- When a manifest exists from Pass 1, bootstrap package overrides from it before exporting in Pass 2.
- Error analysis uses `scripts/analyze_error.py` (LLM via GitHub Models API, `GITHUB_TOKEN` automatically available). Falls back to `scripts/classify_error.py` (regex) if the API call fails — routing is never blocked.
