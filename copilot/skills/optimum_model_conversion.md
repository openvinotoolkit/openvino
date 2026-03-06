# Skill: Model Conversion to OpenVINO IR

**Trigger:** User asks to convert/export a model to OpenVINO format.

## Prerequisites

- Run **optimum_bootstrap** skill first.

## Steps

1. Identify the model ID and pipeline tag (e.g., `text-generation`, `image-text-to-text`).
2. Map the task: `text-generation` → `text-generation-with-past`, `image-text-to-text` → `image-text-to-text`.
3. Run the export:
   ```bash
   optimum-cli export openvino --model <MODEL_ID> --task <TASK> --weight-format fp16 <OUTPUT_DIR>
   ```
4. Verify the output directory contains the expected IR files (`.xml`, `.bin`).
5. Optionally validate with whowhatbench:
   ```bash
   wwb --model <OUTPUT_DIR> --hf <MODEL_ID> --gt <GT_DIR>
   ```
6. Report results: success/failure, output file sizes, any warnings.

## Conventions

- **Task mapping:** `text-generation` → `text-generation-with-past` for decoder models with KV cache.
- **WWB threshold:** Similarity score ≥ 0.9 for passing accuracy validation.
