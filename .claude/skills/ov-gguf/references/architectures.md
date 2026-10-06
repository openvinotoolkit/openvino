# Architecture support

Check [supported_models.md](../../../../src/frontends/gguf/docs/supported_models.md) for the
architecture and checkpoint validation scope before implementing support.

Choose the smallest implementation that matches the reference computation:

- Existing decoder topology: use `make_decoder_architecture()` with the architecture name
  and correct RoPE mode. Supply `DecoderOptions` only for choices that metadata and tensor
  detection cannot resolve. A plain built-in decoder can use a catalog row.
- Different layer order or model family: supply a custom `ModelBuilder`, reusing decoder
  blocks only where their semantics match. Read the relevant sections of
  [the porting guide](../../../../src/frontends/gguf/docs/porting_a_llama_cpp_model.md).
- A component inside an mmproj file: follow [projector support](projectors.md).

Both built-in and external whole-model support use `ArchitectureDefinition`, the same builder
API, converters, and normalization pipeline. Follow the user's chosen delivery route:

- Built-in: read [adding an architecture](../../../../src/frontends/gguf/docs/adding_an_architecture.md)
  and register the definition in the `arch_registry.cpp` catalog.
- External: read [extensions](../../../../src/frontends/gguf/docs/extensions.md) for registration
  and packaging. Register architecture handlers before `load()` and converters/passes before
  `convert()` on the same frontend. Build against the OpenVINO release that will load the library.

Validate the actual shared library when shipping an extension, including
`ov_gguf_architecture_library_tests`. Use numerical fixtures and real-checkpoint comparisons
appropriate to the output: decoder logits through prefill/cached decode, or encoder features.
Check graph fingerprints and rerun the supported architecture regressions after shared-builder
changes. Follow the architecture guide's validation section; record skipped coverage and limits.
