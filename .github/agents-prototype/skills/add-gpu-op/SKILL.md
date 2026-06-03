---
name: add-gpu-op
description: Add a new operation to the OpenVINO GPU plugin — OpenCL kernel design, oneDNN-backed paths, sub-group/LWS tuning, and functional tests.
---

# Skill: Add GPU Op

## When to use
When the Core OpSpec agent has produced a new op spec and you need to implement
GPU plugin support: kernel implementation, registration, and testing.

## Steps

Execute in order — each step produces artifacts consumed by the next.

| Step | File | Purpose |
|---|---|---|
| 0a | [step0-plan.md](step0-plan.md) | Read op spec, build implementation plan, decide kernel vs oneDNN path |
| 0b | [step0-parse-spec.md](step0-parse-spec.md) | Parse op spec JSON/MD into GPU-readable format |
| 1 | [step1-hardware-analysis.md](step1-hardware-analysis.md) | Identify hardware constraints, sub-group size, memory layout requirements |
| 2 | [step2-file-structure.md](step2-file-structure.md) | Create kernel and primitive files, register in factory |
| 3a | [step3-kernel-development.md](step3-kernel-development.md) | Write OpenCL kernel (blocked reads, sub-groups, LWS tuning) |
| 3b | [step3-write-tests.md](step3-write-tests.md) | Write layer tests |
| 3c | [step3-run-tests.md](step3-run-tests.md) | Run and verify tests |
| 3d | [step3-profiling.md](step3-profiling.md) | Profile and tune kernel |
| 4 | [step4-onednn-integration.md](step4-onednn-integration.md) | Add oneDNN-backed primitive path where applicable |
| 5 | [step5-optimize.md](step5-optimize.md) | Final performance optimizations |

## Entry point for orchestrated use

When invoked by the GPU Agent or Enable Operator Agent, start from
[orchestrator.md](orchestrator.md) — it selects the relevant subset of steps
based on the op type and available hardware.
