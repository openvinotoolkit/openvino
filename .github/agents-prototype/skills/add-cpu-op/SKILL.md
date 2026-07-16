---
name: add-cpu-op
description: Add a new operation to the OpenVINO CPU plugin — node registration, JIT/oneDNN executors (AVX2/AVX-512/AMX), and functional tests.
---

# Skill: Add CPU Op

## When to use
When the Core OpSpec agent has produced a new op spec and you need to implement
CPU plugin support: node class, executor strategy, and single-layer tests.

## Steps

Execute in order — each step produces artifacts consumed by the next.

| Step | File | Purpose |
|---|---|---|
| 1 | [step1-analysis.md](step1-analysis.md) | Read op spec, determine implementation strategy (reference / JIT / oneDNN), identify ISA targets and precision requirements |
| 2 | [step2-implementation.md](step2-implementation.md) | Create node class (header + source), register in factory, hook up shape inference |
| 3 | [step3-optimization.md](step3-optimization.md) | Write JIT executor (AVX2/AVX-512/AMX) or oneDNN-backed path; CpuParallel integration |
| 4 | [step4-testing.md](step4-testing.md) | Shared single-layer tests, custom CPU tests, dynamic shapes |
