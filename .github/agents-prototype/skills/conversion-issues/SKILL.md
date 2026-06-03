---
name: conversion-issues
description: Investigate and fix model conversion issues in OpenVINO Frontends (ONNX, PyTorch) — triage, debugging, accuracy comparison, and pre-submission verification.
---

# Agent Skill: Investigate and Fix Frontend Conversion Issues

## Goal
Diagnose and fix issues where models fail to convert to OpenVINO IR or produce incorrect inference results through an OpenVINO frontend.

## Framework-Specific Workflows

Each frontend has its own detailed investigation workflow. Read the one matching the target framework:

| Frontend | Skill file | What it covers |
|---|---|---|
| **ONNX** | [onnx.md](onnx.md) | Triage (unsupported op / conversion bug / shape-type / opset gap), ORT baseline comparison, translator debugging, `.prototxt` test models, C++ GTest, pre-submission checklist |
| **PyTorch** | [pytorch.md](pytorch.md) | Triage (unsupported op / tracing mode / inplace / normalize-step), TorchScript vs torch.export identification, layer test debugging, pre-submission checklist |

## Related Skills (adding new ops)

| Frontend | Skill file | When to use |
|---|---|---|
| ONNX | [add-fe-op/onnx.md](../add-fe-op/onnx.md) | Implementing a new ONNX op translator from scratch |
| PyTorch | [add-fe-op/pytorch.md](../add-fe-op/pytorch.md) | Implementing a new PyTorch op translator from scratch |

## Notes

- Always verify the model works with the framework's reference runtime (ONNX Runtime / PyTorch) before investigating OpenVINO code.
- Prefer minimal, root-cause fixes over broad refactors.
- Every fix needs a test and must pass the full frontend test suite.
