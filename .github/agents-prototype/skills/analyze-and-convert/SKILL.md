---
name: analyze-and-convert
description: Analyze a HuggingFace model and attempt OpenVINO conversion — probe properties, run strategy matrix, classify failures, and produce a structured routing report.
---

# Skill: Analyze and Convert

## When to use
When you have a model ID (or local path) and need to determine whether it converts
to OpenVINO IR, identify why it fails, and emit routing signals for the next agent
in the pipeline.

## Steps

Execute in strict order — each step produces files consumed by the next.

| Step | File | Purpose |
|---|---|---|
| 1 | [probe-model.md](probe-model.md) | Gather model profile (architecture, task, op types) without downloading weights |
| 2 | [try-conversion.md](try-conversion.md) | Run conversion strategy matrix (optimum-cli / ovc / convert_model), capture all outputs |
| 3 | [classify-failure.md](classify-failure.md) | Map errors to taxonomy, extract `missing_op` / `shape_inference` / `accuracy` signals (skip on full success) |
| 4 | [build-report.md](build-report.md) | Assemble structured report, emit `<!-- agent-result -->` marker for orchestrator |
