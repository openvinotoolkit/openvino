---
name: ov-gguf-add-architecture
description: >
  Enable a new model architecture/family in the OpenVINO GGUF frontend through an external
  extension or a built-in builder, or check whether a GGUF model is supported.
  Use when the user asks to enable, support or bring
  up a GGUF/llama.cpp model (llama, qwen, phi, gemma, MoE and similar decoder-only families),
  when a .gguf file is rejected as an unsupported architecture, or when working on
  architecture extensions or arch_registry / DecoderBuilder in src/frontends/gguf/src/builder/.
  Do NOT use for adding
  a single ggml op translator or for debugging wrong output from an already-supported model.
---

1. Check [supported_models.md](../../../src/frontends/gguf/docs/supported_models.md) to see whether the architecture is already accepted and the checkpoint validation scope.
2. Read [extensions.md](../../../src/frontends/gguf/docs/extensions.md) to understand how to write architecture extensions. Choose the route that fits the user's request:

   - **External extension:** provide an `ArchitectureDefinition` through `ArchitectureExtension`. This allows support to be developed and shipped separately without rebuilding OpenVINO.
   - **Built-in support:** register an `ArchitectureDefinition` in `builtin_architectures()` in `arch_registry.cpp`. Read [adding_an_architecture.md](../../../src/frontends/gguf/docs/adding_an_architecture.md) for the builder layout and catalog changes.
   - **mmproj component:** use `ProjectorDefinition` through the derived `ProjectorExtension`. It extends the separate modality/projector-type registry while the mmproj coordinator composes branches. Read [mmproj.md](../../../src/frontends/gguf/docs/mmproj.md) for supported types and component contracts.

   Both whole-model routes use the same definition, builder API, converters, and normalization pipeline. Follow the user's choice; when no route is specified, choose based on whether support should ship separately or as part of OpenVINO.
3. Use the smallest implementation for the chosen route:

   - For an existing decoder topology, use `make_decoder_architecture()` with the architecture name and correct RoPE mode. Supply `DecoderOptions` only for choices that metadata and tensor detection cannot resolve. A plain built-in decoder can use a row in the architecture catalog.
   - For a different layer order or model family, supply a custom `ModelBuilder`. Read [porting_a_llama_cpp_model.md](../../../src/frontends/gguf/docs/porting_a_llama_cpp_model.md) for whole-model examples, shared decoder blocks, and how to move an extension into the built-in catalog.
   - For a new mmproj type, build one encoder/projector branch in the coordinator's `GgufGraphContext`, or reuse `build_builtin_projector()` when metadata and weights fit a supported topology. Return `ProjectorResult`; leave output naming and `finish()` to the coordinator. Use explicit component replacement rather than replacing the whole `clip` handler.
4. For extensions, register architecture and projector handlers before `load()` and any operation converters or transformation passes before `convert()` on the same frontend. Build against the OpenVINO release that will load the extension. Validate the actual shared library if support is shipped that way.
5. Follow the architecture guide's validation steps, including numerical comparisons and graph-fingerprint checks. Re-run other supported architectures after changes to shared builder logic. Successful registration or conversion alone does not establish model accuracy.
