# Adding an op translator to the GGUF frontend

Procedure for enabling a ggml operation. Start with the [two input paths](../README.md),
[tensor/layout contracts](architectures.md#tensor-operations-and-shapes), and
[weight representations](quantization.md). Weight leaves use `GGML_OP_NONE` and supply data
to `translate_weight`; input leaves use the same op name but are resolved to Parameters first.

Related: [architectures.md](architectures.md) (enabling a model family, which
usually needs *no* new op), [debugging_accuracy.md](debugging_accuracy.md) (when a translator
converts but produces wrong numbers).

To supply a converter from an external library without modifying the built-in table, use
[`ConversionExtension`](extensions.md#add-or-override-an-operation-converter). Register it on the
same frontend as any architecture or projector extension before conversion.

## Before writing a translator

Check that an op translator is actually what is missing:

- **A new architecture or projector** needs an appropriate [builder registration](extensions.md#choose-an-extension).
  Add a translator only when its computation needs an operation missing from the table.
- **A structurally different use of an existing op** is a new named attribute that describes the
  operation (for example `view_slice`, `cont_reshape`, `rope_mode`), not a new translator. Read the
  [tensor operation contracts](architectures.md#tensor-operations-and-shapes) first: an attribute
  describes the operation's geometry or parameters, never the decoder that produced it.
- Both decoder paths (native builder and llama.cpp cgraph) share translator bodies, so a change here
  affects both. Keep the body path-agnostic: read operand shapes and named attributes that either
  decoder can supply, never branch on "which decoder made this".

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

    std::shared_ptr<ov::Node> res = ...;

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}
```

Always finish with `rename_outputs_with_suffix(..., context.get_name())`: the walk stores results in
the `TensorMap` under the decoder's output names, and stable friendly names are what the passes and
the graph-fingerprint gate rely on.

A ggml node has one output. A native builder may declare more with `add_op(..., extra_outputs)`;
the translator then returns one output per name, in order. `GATED_DELTA_NET` uses this
(`split_outputs`) to return the attention and the new recurrent state separately instead of
ggml's packed tensor.

`NodeContext` ([`src/node_context.hpp`](../src/node_context.hpp)):

| Call | Purpose |
|------|---------|
| `get_input(idx)` / `get_input(name)` | Operand as `Output<Node>` |
| `has_input(name)` | Test an optional operand first |
| `get_input_size()` | Actual operand count |
| `get_input(idx).get_partial_shape()` | Shape inferred by OpenVINO, including dynamic dimensions |
| `get_input_shape(idx)` | Logical operand shape; native builders return the connected value's shape and cgraph decoders return GGML `ne[]` in OpenVINO order |
| `get_output_shape()` | GGML destination metadata for operations such as raw strided VIEW; unavailable for native builder operations |
| `get_input_view_element_offset(idx)` | Element (not byte) offset for a ggml VIEW operand |
| `get_attribute<T>(name[, default])` | Any other typed op parameter |

Helpers in [`src/utils.hpp`](../src/utils.hpp): `num_inputs_check`, `get_dimensions`,
`rename_outputs_with_suffix` and `make_sin_cos` (RoPE).

`TOP_K` accepts an `int64_t` attribute `k`; OpenVINO infers the output shape from the input
and this parameter. Every producer must supply `k`; captured output metadata is not an operation
descriptor.

`REPEAT` accepts fixed integer `repeats`. If a repeat factor varies at runtime, supply the target
tensor as its second logical operand; the translator derives the factor from both operand shapes.

`ROPE` accepts an optional `int64_t` attribute `rope_offset` (default zero), matching
`ggml_rope_set_offset`: rotate `n_dims` channels starting at that even offset and preserve
the prefix and tail of each head.

`FLASH_ATTN_EXT` accepts an optional boolean `sink_without_mask` attribute for a decoder
that compacts null ggml sources: set it when `src[3]` is null and `src[4]` contains
attention sinks, leaving four input tensors. The cgraph convention of naming that
weight `*.attn_sinks.weight` is also recognized. Otherwise the fourth tensor is
treated as an attention mask; its shape and element type cannot distinguish a
mask from sinks (an F32 mask can have the same shape as a sink).

Converters must infer intermediate shapes from their OpenVINO operands, reading only the axes
needed for the operation. A dynamic token axis does not prevent reading a static head width.
Use explicit attributes for operation parameters: `reshape_target` / `special_zero`, `view_slice`,
`perm`, `repeats`, and TopK `k`. Do not use destination metadata as an implicit reshape pattern or
repair an operand based on its producer. A new converter also serves `GgufGraphContext::node`
without a separate shape implementation in the builder.

Do not add model inputs for values already available from operands: use `ShapeOf` for runtime
extents and arithmetic for dependent bounds (for example, sequence end = start + count). Use
constants only for values fixed for the lifetime of the compiled graph. Independent runtime
information, such as recurrent reset ranges and writeback offsets, remains explicit Parameters,
including for static-shape models. Bind those Parameters from the current compute state, not
from a second cached input-value map.

### Graph-ingestion attributes

The frontend owns lowering for both ingestion paths. The llama.cpp adapter exposes facts through
the existing decoder interface; it must not construct operation subgraphs or reinterpret frontend
case numbers. The following attributes supplement facts that are not encoded as logical operands:

- `rope_mode`: `normal`, `neox`, `imrope`, `mrope` or `vision`, ggml's rotation semantics. Its operand is an
  already-materialized logical tensor. Sectioned modes read `RopeConfig::sections`.
- `GATED_DELTA_NET` takes an optional seventh operand, the number of valid tokens in a padded
  static chunk; padded tokens then leave the recurrent state unchanged.
- `has_mask`: whether FLASH_ATTN_EXT has a mask. An absent mask and present sinks otherwise look
  like a four-input masked operation after null source slots are removed.
- `max_bias` and `kq_soft_cap`: separate FLASH_ATTN_EXT parameters, not interchangeable.
  Nonzero `max_bias` (ALiBi) scales the mask by ggml's per-head slopes and requires a mask.
- `kv_view`: a VIEW that reads a KV cache whose live length and first stream move with the batch.
  The descriptor is `{streams, row_width, heads, head_size, active_streams, row_base, transposed,
  order[4], static_length}`; optional operands after the cache are the live length and the first
  stream. `transposed` selects the cells-innermost (`-fa off` V) layout, and `order` places the
  canonical (stream, cell, head, dim) axes in the view's OV order.
- `output_dynamic_axis`: OV-order axis for a runtime-sized raw VIEW output; `-1` means a fixed
  extent. The cgraph decoder also uses it when producing an explicit `cont_reshape` descriptor.
  Do not infer dynamic intent just because two captured dimensions happen to be equal.

VIEW's `view_input_strides` and `view_output_strides` are byte strides in GGML order;
`view_offset_bytes` is relative to the immediate logical source, not the storage root.
The frontend resolves supported dense/rectangular views before consumers run. PERMUTE uses only
the OV-order `perm` attribute and the logical operand it receives. Unsupported view layouts are
errors.

SET_ROWS updates the destination name in the translation tensor map. Later readers therefore
observe the new value, while earlier readers retain their original value. A CPY update (`axis`
present; destination is operand 1) and a ranged SCALE (start/length operands; the tensor is
operand 0) return their whole destination and update its storage the same way, which is also what
the model output named after that storage reports. This is graph conversion
semantics, not runtime buffer ownership; the host still owns execution and buffer lifetime.
Full-storage RESHAPE aliases are tracked by source tensor names, not by OpenVINO node identity:
a CONT copy may lower to the same node but is not a storage alias. Alias shape expressions are
replayed on the updated cache so they do not retain stale extents when stateful conversion grows it.

Contract tests must reuse a compiled request across changing token counts and compare values and
state updates, not merely inspect successful conversion. The synthetic decoder tests exercise the
frontend interface but do not replace live llama.cpp backend tests. The native-file acceptance gate
is `GGUFArchitectureAccuracy.PrefillAndCachedDecodeMatchLlamaCPU`; its numerical fixtures and a
matched pure-ggml CPU reference are required before claiming end-to-end correctness.

Result-type rules belong in the converter, not in the builder or architecture. `MUL_MAT` and
`MUL_MAT_ID` produce F32, `TOP_K`/`ARGSORT` produce I32, and `GET_ROWS` returns I32 for integer
rows and F32 for floating/quantized rows. Type-preserving operations use their input type;
`SET`, `SET_ROWS` and `CPY` use the destination operand. Construct the OpenVINO computation in
the appropriate precision; casting an already-overflowed F16 matrix product to F32 is too late.

Only operations with an explicit destination-type parameter accept `dst_type`: a one-input `CPY`
represents a cast, and `IM2COL` takes the type chosen by the caller. Those converters also accept
legacy cgraph `output_type` metadata when no destination type can otherwise be determined.
Native graph nodes have no output-type field or default-type table. A new converter derives its
own result type from operation semantics, inputs and attributes.

Prefer `ov::op::vX::OpName` over `opsetX::OpName`, per the repository convention.

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

Use the [reference policy](debugging_accuracy.md#reference-and-precision):

- Simple elementwise ops with an unambiguous closed form — compute inline in the test.
- Anything with layout, geometry or head structure (rope, conv, attention, views) — generate the
  reference from ggml-CPU with a standalone oracle such as `ssm_conv_oracle.c` / `imrope_oracle.c`
  (see [the ggml-CPU oracle](debugging_accuracy.md#the-ggml-cpu-oracle)). Commit the oracle with the
  test and name it in a comment next to the pasted or `.npy` expectations.

Test at realistic dimensions. With one head many layout orders coincide, so a single-head test can
pass against a wrong reference.

`expect_near(actual, expected, atol = 1e-4f, rtol = 2e-3f)` combines absolute and relative
tolerance. **Do not tighten `rtol`**: ARM CPU runs the graph in fp16 by default, and a tolerance
tuned only on fp32 x86 will fail there.

To find the closest existing example without reading the whole (large) file:

```bash
rg -n '^TEST\(' src/frontends/gguf/tests/test_ops.cpp
```

## Build and run

Follow [testing.md](testing.md) for build flags, binary paths, filters, and CI coverage.
Run the frontend binary unfiltered before finishing so the operation-coverage gate executes.
Shared translator or VIEW/attribute-contract changes also need architecture fingerprints and numerical
regressions: a guard that fixes one model can reject another's legitimately contiguous view.

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
