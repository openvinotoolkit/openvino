# Choosing an Implementation Strategy for a New CPU Op

This is the analysis phase of adding an operation to the Intel CPU plugin. Before
writing any code, locate the op's contract and its reference implementation,
check whether a CPU node already exists, and decide which implementation strategy
to follow.

## Contents

- [Locating the core op and its reference](#locating-the-core-op-and-its-reference)
- [Does a CPU node already exist?](#does-a-cpu-node-already-exist)
- [The strategy decision matrix](#the-strategy-decision-matrix)
- [Supported precisions and layouts](#supported-precisions-and-layouts)
- [CPU-specific transformations](#cpu-specific-transformations)

## Locating the core op and its reference

Find the core op class to understand the operation's contract:

```bash
# Find the op header
find src/core/include/openvino/op/ -iname "*<op_name>*"

# Check which opset it belongs to
grep -r "<OpName>" src/core/include/openvino/opsets/
```

Read the op header to extract:
- **Inputs**: count, element types, shape constraints.
- **Outputs**: count, element types, shape inference rules.
- **Attributes**: names, types, defaults.
- **Base class**: `Op`, `util::UnaryElementwiseArithmetic`, etc.

The reference implementation may reside in one of two places:

**a) Standalone reference function:**
```bash
find src/core/reference/include/openvino/reference/ -iname "*<op_name>*"
```

**b) Inside the op's `evaluate()` method:**
Check the op header or source — many ops implement `evaluate()` directly,
which contains the reference algorithm. This is the method called by the
`Reference` fallback node.

```bash
grep -n "evaluate" src/core/include/openvino/op/<op_name>.hpp
grep -rn "evaluate" src/core/src/op/<op_name>.cpp
```

Read the reference implementation (whichever location) to understand:
- Algorithm complexity (element-wise, reduction, convolution-like, etc.).
- Data type handling (templated on element type or generic).
- Whether it supports dynamic output shapes.

## Does a CPU node already exist?

```bash
# Check if a node exists
find src/plugins/intel_cpu/src/nodes/ -iname "*<op_name>*"

# Check Type enum
grep -n "<OpName>\|<op_name>" src/plugins/intel_cpu/src/cpu_types.h

# Check factory registration
grep -n "<OpName>\|<op_name>" src/plugins/intel_cpu/src/nodes_factory.cpp
```

If a dedicated node **already exists**, determine what needs updating:
- Missing precision support?
- Missing dynamic shape support?
- Performance issue requiring JIT optimisation?

If **no dedicated node exists**, the op currently runs through the `Reference`
fallback node (if it has `evaluate()` implemented). However, not all reference
implementations are compiled into the shared libraries — some may be excluded
by selective build (see [Selective build](../selective_build.md)). A dedicated
CPU node should implement its own optimised C++ (cache-aware memory access,
`CpuParallel` parallelisation, function inlining). The core reference (from
`ov::reference::` or the op's `evaluate()` method) should be reused only if it
already meets these performance criteria.

## The strategy decision matrix

Choose one of these approaches based on the op characteristics:

| Strategy | When to Use | Example Ops |
|----------|-------------|-------------|
| **Reference-only** | Simple ops where `Reference` fallback is sufficient; op is rarely used or not perf-critical | `Eye`, `Bucketize` |
| **Portable C++ with CpuParallel** | Trivially parallelisable, no SIMD benefit, moderate perf needs. Directly in the node's `execute()` method. | `SegmentMax`, `SearchSorted` |
| **Executor-based** | **Default for any op more complex than portable C++.** Must be used whenever JIT kernels, oneDNN primitives, ISA-specific paths (x64/ARM/RISC-V), or multiple backend implementations are involved. | `FullyConnected`, `Convolution`, `Eltwise`, `Interpolate`, `Reduce`, `Pooling` |

> **Key rule:** The executor-based approach is the standard architecture for new
> CPU nodes. Even JIT-optimised and oneDNN-backed ops use the executor framework
> — the JIT kernel becomes an `ExecutorType::Jit` implementation, the oneDNN
> primitive becomes an `ExecutorType::Dnnl` implementation, and a portable C++
> reference becomes an `ExecutorType::Reference` fallback. Study
> `eltwise_implementations.cpp`, `fullyconnected_implementations.cpp`, and
> `convolution_implementations.cpp` as canonical examples.

Decision criteria:
1. **Is the op trivially simple and not perf-critical?** → Portable C++ in `execute()` with `CpuParallel` may suffice (no executor needed).
2. **Does the op benefit from JIT / SIMD / oneDNN / vendor libraries?** → Must use executor framework. JIT kernels, oneDNN primitives, and ACL/MLAS are registered as separate `ExecutorImplementation` entries.
3. **Does the op support multiple ISA or platform targets?** → Must use executor framework so it can automatically choose between JIT/Dnnl/ACL/Reference.
4. **Is dynamic output shape data-dependent?** → Needs custom `needShapeInfer()`.
5. **Do output shapes depend on the operation results?** → Use `InternalDynShapeInferFactory()` and update shape in `execute()`.

The chosen strategy determines how much of the rest applies: a reference-only or
portable-C++ op needs only the node itself
([Implementing a CPU Node](./implementing_a_node.md)); an executor-based op also
uses the executor framework
([Executors, Kernels and Optimization](./executors_and_optimization.md)).

## Supported precisions and layouts

Standard precision support matrix:

| Precision | Common Support | Notes |
|-----------|---------------|-------|
| `f32` | Always | Baseline precision |
| `bf16` | If ISA >= AVX-512 BF16 | Check `mayiuse(avx512_core_bf16)` |
| `f16` | If ISA >= AVX-512 FP16 | Check `mayiuse(avx512_core_fp16)` |
| `i8` / `u8` | For quantized ops | INT8 path |
| `i32` / `i64` | For index/integer ops | Index types |

Standard layout types for `addSupportedPrimDesc`:

| Layout | Constant | Use Case |
|--------|----------|----------|
| Planar (NCHW) | `LayoutType::ncsp` | Default, always supported |
| Channels-last (NHWC) | `LayoutType::nspc` | Conv, Pooling adjacency |

## CPU-specific transformations

```bash
grep -r "<OpName>" src/plugins/intel_cpu/src/transformations/
```

Some ops are decomposed or fused by CPU-specific transformations before reaching
the node layer. Understand these to avoid duplicating logic. For the graph-level
optimizations that may consume or rewrite the op (fusing into convolutions,
FakeQuantize handling, etc.), see
[Internal CPU Plugin Optimizations](../internal_cpu_plugin_optimization.md).
