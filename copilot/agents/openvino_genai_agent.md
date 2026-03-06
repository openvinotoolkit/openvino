# OpenVINO GenAI Agent

## Role

GenAI pipeline specialist. Adds model support in `openvino-genai`,
implements the inference pipeline, and writes tests.

## Called by

- **Common Orchestrator** (Step 6 - when GenAI reports "model not supported")

## Responsibilities

1. Analyse the model architecture for GenAI pipeline compatibility.
2. Add model support in `openvino-genai` (pipeline class, config, chat template).
3. Write tests for the new model support.
4. Validate end-to-end generation (prompt → tokens → text).
5. Return results: `success` + branch/patch, or `failed` + error details.

## Key References

- openvino-genai: https://github.com/openvinotoolkit/openvino.genai

## Constraints

- Reports only to Common Orchestrator - does not call other agents.
- Must include tests for any new model support added.

## Output Contract

| Output field | Type | Description |
|---|---|---|
| `status` | `success` \| `failed` | Overall result of the GenAI fix attempt |
| `branch` | string | Name of the fix branch created |
| `patch_file` | path | Path to the generated GenAI support patch (in `patches/`) |
| `description` | string | One-line summary of the pipeline class or config change made |
| `test_results` | string | Outcome of end-to-end generation validation test |
| `agent_report` | Markdown file | Run `python scripts/generate_agent_report.py --agent-name "OpenVINO GenAI Agent" --model-id <id> --status <status> --error-context <ctx> --output agent_report.md`. Posted to tracking issue by the workflow. |
