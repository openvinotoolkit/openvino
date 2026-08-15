---
name: ov-gguf-add-architecture
description: >
  Enable a new model architecture/family in the OpenVINO GGUF frontend's native .gguf builder,
  or check whether a GGUF model is supported. Use when the user asks to enable, support or bring
  up a GGUF/llama.cpp model (llama, qwen, phi, gemma, MoE and similar decoder-only families),
  when a .gguf file is rejected as an unsupported architecture, or when working on
  arch_registry / DecoderBuilder in src/frontends/gguf/src/builder/. Do NOT use for adding
  a single ggml op translator or for debugging wrong output from an already-supported model.
---

1. Read [src/frontends/gguf/docs/adding_an_architecture.md](../../../src/frontends/gguf/docs/adding_an_architecture.md) — most architectures in the transformer family need **no code**, only a `general.architecture` string added to `arch_registry.cpp` plus the correct RoPE type. It also lists exactly which structural features are auto-detected from the tensor table and metadata, and how to verify a new arch.
2. Check [src/frontends/gguf/docs/supported_models.md](../../../src/frontends/gguf/docs/supported_models.md) first to see whether the architecture is already accepted and how support was verified.
3. Only if the family is structurally novel, consult the "10% case" section and [frontend_design.md](../../../src/frontends/gguf/docs/frontend_design.md) before adding builder code.
4. Verify as that document prescribes, including the graph-fingerprint check, and re-run the other supported architectures after any change to shared builder logic.
