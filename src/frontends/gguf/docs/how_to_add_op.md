# Adding an op translator to the GGUF frontend

Procedure for enabling a ggml operation. For the *concepts* behind it — the two decoder paths, the
`GGML_OP_NONE` weight convention, `op_case` numbering, the memory model — read
[frontend_design.md](frontend_design.md) first; this document does not repeat them.

Related: [adding_an_architecture.md](adding_an_architecture.md) (enabling a model family, which
usually needs *no* new op), [debugging_accuracy.md](debugging_accuracy.md) (when a translator
converts but produces wrong numbers).

## Before writing a translator

Check that an op translator is actually what is missing:

- **A new model family** normally needs only an entry in `supported_archs()` — see
  [adding_an_architecture.md](adding_an_architecture.md). Reach for a translator only when the graph
  genuinely contains a ggml op the table does not have.
- **A structurally different use of an existing op** is an `op_case`, not a new translator. Read the
  `op_case` section of [frontend_design.md](frontend_design.md) before adding a case — a case that
  exists only to mean "this came from the builder" is a defect.
- Both decoder paths (native builder and llama.cpp cgraph) share translator bodies, so a change here
  affects both. Keep the body path-agnostic; branch on `op_case`, never on "which decoder made this".

## Checklist

| # | File | Change |
|---|------|--------|
| 1 | `src/op/<name>.cpp` | New translator function |
| 2 | [`src/op_table.hpp`](../src/op_table.hpp) | `GGUF_OP_CONVERTER(translate_<name>);` declaration |
| 3 | [`src/op_table.cpp`](../src/op_table.cpp) | `{"GGML_OP_<NAME>", op::translate_<name>},` (list is alphabetical) |
| 4 | [`tests/CMakeLists.txt`](../tests/CMakeLists.txt) | Add `"${FE_SRC_DIR}/op/<name>.cpp"` to `FRONTEND_SRCS` |
| 5 | [`tests/test_ops.cpp`](../tests/test_ops.cpp) | `TEST(GGUFOps, <Name>)` — **mandatory**, see the coverage gate below |

> **Step 4 is mandatory and easy to miss.** `src/CMakeLists.txt` builds the library via
> `ov_add_frontend`, which picks up new sources automatically — but the test binary compiles the
> frontend sources from an **explicit list with no GLOB**. Omitting this yields an undefined symbol
> at link time, after a full compile.

If the op maps 1:1 onto a single OpenVINO op with the same operand order, skip steps 1, 2 and 4 and
register a template from [`src/utils.hpp`](../src/utils.hpp) directly in `op_table.cpp`:

```cpp
{"GGML_OP_SUB", op::translate_1to1_match_2_inputs<v1::Subtract>},
{"GGML_UNARY_OP_TANH", op::translate_1to1_match_1_input<v0::Tanh>},
```

## Translator shape

```cpp
OutputVector translate_<name>(const NodeContext& context) {
    num_inputs_check(context, 1, 2);        // min / max operand count

    float eps = context.get_attribute<float>("eps");
    int op_case = context.get_op_case();

    std::shared_ptr<ov::Node> res = ...;

    return rename_outputs_with_suffix({res}, context.get_name());
}
```

Always finish with `rename_outputs_with_suffix(..., context.get_name())`: the walk stores results in
the `TensorMap` under the decoder's output names, and stable friendly names are what the passes and
the graph-fingerprint gate rely on.

`NodeContext` ([`src/node_context.hpp`](../src/node_context.hpp)):

| Call | Purpose |
|------|---------|
| `get_input(idx)` / `get_input(name)` | Operand as `Output<Node>` |
| `has_input(name)` | Test an optional operand first |
| `get_input_size()` | Actual operand count |
| `get_input_shape(idx)` / `get_output_shape()` | **Static ggml** shape — use when the live OV shape is dynamic (KV-cache path) |
| `get_input_view_element_offset(idx)` | Element (not byte) offset for a ggml VIEW operand |
| `get_op_case()` | Structural variant (convenience wrapper, defaults to 0) |
| `get_output_type()` | Declared output element type |
| `get_attribute<T>(name[, default])` | Any other typed op parameter |

Helpers in [`src/utils.hpp`](../src/utils.hpp): `num_inputs_check`, `get_dimensions`,
`rename_outputs_with_suffix`, `make_sin_cos` (RoPE), `process_view_input`.

Insert a `Convert` to `get_output_type()` when the op may change element type (`CONCAT`, `CPY`,
`SET_ROWS`, `GET_ROWS`) rather than assuming the input type. Prefer `ov::op::vX::OpName` over
`opsetX::OpName`, per the repository convention.

## Test, and the coverage gate

[`tests/test_op_coverage.cpp`](../tests/test_op_coverage.cpp) asserts that **every op registered in
`op_table.cpp` is exercised by some test**. Registering a translator without a test fails the suite;
the gate exists because `GGML_UNARY_OP_GELU_QUICK` once shipped with the wrong formula precisely
because nothing converted it. The exemption list is for ops whose *nature* makes a single-op test
meaningless — not for ops that are merely awkward to test.

The gate only asserts when the full suite runs, so a narrowing `--gtest_filter` silently skips it.
**Run the binary unfiltered before pushing.**

```cpp
TEST(GGUFOps, Scale) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SCALE")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<float>("scale", 2.5f)
                     .attr<float>("bias", 1.0f)
                     .build();

    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});
    expect_near(out, expected);
}
```

`SingleOpBuilder` drives the real `FrontEnd::convert` through an in-memory `SingleOpDecoder`, so no
`.gguf` file is involved. Helpers are in [`op_test_utils.hpp`](../tests/op_test_utils.hpp).

**Where the expected values come from matters more than the test's shape.** Per the one rule in
[debugging_accuracy.md](debugging_accuracy.md), the reference must come from real ggml, not from
your own reading of the op's math:

- Simple elementwise ops with an unambiguous closed form — compute inline in the test.
- Anything with layout, geometry or head structure (rope, conv, attention, views) — generate the
  reference from ggml-CPU: an `.npy` fixture via [`gen_ggml_reference.c`](../tests/gen_ggml_reference.c),
  or a standalone oracle such as `ssm_conv_oracle.c` / `imrope_oracle.c`, and paste its output with a
  comment naming the oracle.

Test at realistic dimensions. With one head many layout orders coincide, so a single-head test can
pass against a wrong reference.

`expect_near(actual, expected, atol = 1e-4f, rtol = 2e-3f)` combines absolute and relative
tolerance. **Do not tighten `rtol`**: ARM CPU runs the graph in fp16 by default, and a tolerance
tuned only on fp32 x86 will fail there.

To find the closest existing example without reading the whole (large) file:

```bash
grep -n "^TEST(" src/frontends/gguf/tests/test_ops.cpp
```

## Build and run

The frontend is **off by default** (`ENABLE_OV_GGUF_FRONTEND` in
[`cmake/features.cmake`](../../../../cmake/features.cmake)); without it the test target does not
exist.

```bash
cmake -B build -DENABLE_OV_GGUF_FRONTEND=ON -DENABLE_TESTS=ON
cmake --build build --target ov_gguf_frontend_tests -j$(nproc)

# iterate on one op ...
./build/bin/*/*/ov_gguf_frontend_tests --gtest_filter='GGUFOps.<Name>*'
# ... then unfiltered, so the coverage gate actually runs
./build/bin/*/*/ov_gguf_frontend_tests
```

CI runs the same binary in the "GGUF frontend tests" step of
[`job_cxx_unit_tests.yml`](../../../../.github/workflows/job_cxx_unit_tests.yml), ungated by Smart CI.

For a change that touches a shared translator or a VIEW/`op_case` predicate, also re-run the
graph-fingerprint check ([`tests/graph_fingerprint.py`](../tests/graph_fingerprint.py)) across the
supported architectures: a guard that fixes one arch can reject another's legitimately-contiguous
view.

## Bringing up a model that hits a missing op

Conversion aborts on the **first** unsupported op with:

```
Translation for operation type GGML_OP_<NAME> is not implemented.
```

Enabling one op per run is the slow path. Instead diff the model's op vocabulary against the keys of
`get_supported_ops()` to get the full set at once, group the ops that collapse into
`translate_1to1_match_*` one-liners, and implement only the remainder as real translators.

A different error, `Number of <op> outputs greater than number of converted outputs`, means the
translator returned the wrong number of outputs — not that the op is unsupported.
