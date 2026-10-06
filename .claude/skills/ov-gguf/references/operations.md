# Operation translation

Confirm the missing layer: an unclaimed architecture needs a builder registration; a new
mmproj type needs a projector component; a structurally different use of an existing operation
may need a semantic `op_case`. Both native and supplied decoders share converters.

For a built-in converter, read
[how to add an op](../../../../src/frontends/gguf/docs/how_to_add_op.md) for the five-file
checklist, `NodeContext`, and coverage gate. The test CMake source list is explicit: add the
translator there as well as to the library's registration table. For layout and case contracts,
use [tensor operations and shapes](../../../../src/frontends/gguf/docs/porting_a_llama_cpp_model.md#tensor-operations-and-shapes).
An external converter instead uses
[ConversionExtension](../../../../src/frontends/gguf/docs/extensions.md#add-or-override-an-operation-converter)
on the same frontend before `convert()`.

Find nearby examples with `rg -n 'GGML_' src/frontends/gguf/src/op_table.cpp` and
`rg -n '^TEST\(' src/frontends/gguf/tests/test_ops.cpp`, then read the relevant ranges.
Derive result shapes/types from operands and semantic attributes; keep translator bodies
independent of the decoder path.

Generate layout-sensitive numerical expectations from real ggml CPU, using nontrivial head,
token, and batch dimensions. Simple elementwise checks may use unambiguous closed forms.
Cover relevant semantic cases and fused/fallback paths. Preserve tolerances needed by supported
platforms rather than tightening them from an x86-only run.

Build `ov_gguf_frontend_tests` with `ENABLE_OV_GGUF_FRONTEND=ON`, `ENABLE_TESTS=ON`, and
`BUILD_SHARED_LIBS=ON`; use the configured runtime output directory to locate the binary.
Filter while iterating, then run it unfiltered so the op-coverage gate executes. Shared
translator or VIEW/case changes also require architecture fingerprints and numerical regressions.
