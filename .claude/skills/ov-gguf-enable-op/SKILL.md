---
name: ov-gguf-enable-op
description: >
  Enable a ggml operation in the OpenVINO GGUF frontend by adding or fixing an op translator.
  Use when conversion fails with "Translation for operation type GGML_OP_* is not implemented",
  when the user asks to add/implement/fix a GGUF or ggml op translator or an op_case, or when
  working on src/frontends/gguf/src/op/. Do NOT use for enabling a new model architecture
  (that is usually just a name in arch_registry.cpp), for wrong numerical output from an op that
  already converts, or for GGUF quantization format work.
---

1. Read [src/frontends/gguf/docs/how_to_add_op.md](../../../src/frontends/gguf/docs/how_to_add_op.md) — the five-file checklist (including the test CMake source list, which is not globbed), the `NodeContext` API, the mandatory op-coverage gate, where reference values must come from, and the build/test commands (`-DENABLE_OV_GGUF_FRONTEND=ON`).
2. Confirm a translator is really what is missing: a new architecture usually needs only a name in `arch_registry.cpp`, and a structurally different use of an existing op is an `op_case`. Both are covered in that document.
3. Find the closest existing op with `grep -n "GGML_" src/frontends/gguf/src/op_table.cpp` and the closest test with `grep -n "^TEST(" src/frontends/gguf/tests/test_ops.cpp`, then read only those ranges — do not read `test_ops.cpp` in full.
4. Implement, then build and run `ov_gguf_frontend_tests` — filtered while iterating, unfiltered before finishing so the coverage gate runs.
