# Executors, Kernels and Optimization

This is the optimization phase of adding an operation to the Intel CPU plugin. It
covers the executor framework — the standard architecture for any op more complex
than portable C++ — and the optimization techniques that live **inside** executor
implementations: `CpuParallel` multithreading, JIT kernels with ISA-specific
paths, oneDNN-backed primitives, and eltwise JIT emitters.

It assumes the node compiles and runs via a reference path (see
[Implementing a CPU Node](./implementing_a_node.md)) and that the ISA targets and
strategy were chosen during
[analysis](./choosing_a_strategy.md#the-strategy-decision-matrix).

## Contents

- [When to optimize (and when to skip)](#when-to-optimize-and-when-to-skip)
- [The executor framework architecture](#the-executor-framework-architecture)
- [Defining Attrs/Config and getImplementations](#defining-attrsconfig-and-getimplementations)
- [Implementing concrete executors](#implementing-concrete-executors)
- [Executor framework utilities](#executor-framework-utilities)
- [CpuParallel multithreading](#cpuparallel-multithreading)
- [Writing a JIT kernel and JIT executor](#writing-a-jit-kernel-and-jit-executor)
- [oneDNN executor implementations](#onednn-executor-implementations)
- [Eltwise JIT emitters](#eltwise-jit-emitters)
- [Functional verification](#functional-verification)

## When to optimize (and when to skip)

Optimization is **mandatory** for performance-critical ops (element-wise,
reductions, attention, normalization). It may be **skipped** for simple ops where
the reference implementation is sufficient — mark the phase as `skipped`; the
overall result is still success if implementation and tests pass.

Skip criteria:
- Op is rarely used in production models.
- Reference implementation already meets performance requirements.
- Op is pure data movement (handled efficiently by `Reorder` or `Reshape`).

All optimization techniques below are used **inside executor implementations**.
The executor framework handles ISA dispatch, fallback ordering, and precision
matching; this doc focuses on the code that goes inside each executor class.

## The executor framework architecture

The executor framework is the **standard architecture** for any CPU node that is
more complex than simple portable C++ code. This includes ops with:
- JIT kernels (these become `ExecutorType::Jit` implementations)
- oneDNN primitives (these become `ExecutorType::Dnnl` implementations)
- Multiple backend targets (x64/ARM/RISC-V)
- Any ISA-specific optimisation

Study these canonical implementations in this order:
1. `eltwise_implementations.cpp` — simplest executor pattern (Jit + ACL + Reference)
2. `fullyconnected_implementations.cpp` — full-featured (MLAS, 1x1conv, ACL, KleidiAI, oneDNN)
3. `convolution_implementations.cpp` — layout-aware implementations with `MemoryFormatFilter`

```
Node (e.g. Eltwise, FullyConnected, Convolution)
 ├── OpNameAttrs           (op config struct: <op_name>_config.hpp)
 ├── MemoryArgs memory     (maps ARG_SRC/ARG_DST/... → MemoryPtr)
 ├── ExecutorFactoryPtr<OpNameAttrs> factory
 │    ├── filters ExecutorImplementation<Attrs> list by:
 │    │    ├── supports(config, memoryFormatFilter)   — ISA, precision, layout
 │    │    ├── createOptimalConfig(config)             — preferred memory format
 │    │    └── acceptsShape(attrs, memory)             — shape-dependent heuristics
 │    └── make(memory) → returns ExecutorPtr or VariableExecutor
 └── ExecutorPtr executor
      ├── update(memory)   — called in prepareParams()
      └── execute(memory)  — called in execute()
```

## Defining Attrs/Config and getImplementations

### Step 1: Define the Attrs struct

Create `src/plugins/intel_cpu/src/nodes/executors/<op_name>_config.hpp`:

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"
#include "post_ops.hpp"  // if post-ops fusion is supported

namespace ov::intel_cpu {

struct OpNameAttrs {
    // Op-specific attributes needed by all implementations.
    // Examples: float epsilon; int axis; bool keep_dims;
    PostOps postOps;  // include if the op supports post-op fusion
};

using OpNameConfig = executor::Config<OpNameAttrs>;

}  // namespace ov::intel_cpu
```

### Step 2: Register the `getImplementations` specialisation

**a) Declare in `implementations.hpp`** — add alongside existing declarations:

```cpp
// OpName
template <>
const std::vector<ExecutorImplementation<OpNameAttrs>>& getImplementations();
```

**b) Define in `<op_name>_implementations.cpp`:**

The file registers all implementation variants in **priority order** (highest
first). Each entry uses an `OV_CPU_INSTANCE_*` macro that conditionally compiles
the implementation based on the target architecture:

| Macro | Active When |
|-------|-------------|
| `OV_CPU_INSTANCE_COMMON(...)` | Always (portable code) |
| `OV_CPU_INSTANCE_X64(...)` | x86-64 only |
| `OV_CPU_INSTANCE_DNNL_X64(...)` | x86-64 with oneDNN |
| `OV_CPU_INSTANCE_DNNL(...)` | Any arch with oneDNN |
| `OV_CPU_INSTANCE_ACL(...)` | AArch64 with ARM Compute Library |
| `OV_CPU_INSTANCE_KLEIDIAI(...)` | AArch64 with KleidiAI |
| `OV_CPU_INSTANCE_MLAS_X64(...)` | x86-64 with MLAS |
| `OV_CPU_INSTANCE_RISCV64(...)` | RISC-V 64 only |

These macros (defined in `utils/arch_macros.h`) compile code for unavailable
platforms away, supporting the conditional-compilation infrastructure (see
[Selective build](../selective_build.md)).

Each `ExecutorImplementation<Attrs>` has these fields:
1. **name** — unique human-readable string (e.g. `"op_name_jit_ncsp"`)
2. **ExecutorType** — `Jit`, `Dnnl`, `Acl`, `Reference`, etc.
3. **OperationType** — `OperationType::OpName`
4. **supports** — lambda returning `bool`. Use `VERIFY(condition, MESSAGE)` macro
   for debuggable rejection. Can take `(const Config&)` or
   `(const Config&, const MemoryFormatFilter&)`.
5. **createOptimalConfig** — lambda returning `std::optional<Config>`. Return
   `{}` (nullopt) if the current memory config is already acceptable. Use
   `createOptimalConfigCommon()` helper with `TypeMapping` and `LayoutConfig`.
6. **acceptsShape** — lambda `(const Attrs&, const MemoryArgs&) -> bool`. Pass
   `AcceptsAnyShape<Attrs>` (or `{}`) for shape-agnostic implementations.
7. **create** — lambda returning `ExecutorPtr`. Use `CreateDefault<Executor, Attrs>{}`
   for simple cases or a custom lambda for complex wiring (e.g., `DnnlExecutor`).

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/<op_name>_config.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "utils/arch_macros.h"

// Include the actual executor headers:
#include "nodes/executors/jit/<op_name>_jit.hpp"      // JIT executor (x64)
// #include "nodes/executors/acl/<op_name>_acl.hpp"    // ACL executor (ARM)
#include "nodes/executors/ref/<op_name>_ref.hpp"       // Reference executor

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;

// TypeMapping defines precision transformations per implementation.
// Each row: {input_precisions..., output_precisions...} → {transform_functions...}
// Transform helpers: bypass() = keep as-is, just<T>() = force to T, use<N>() = match arg N
static const TypeMapping opNameTypeMapping {
    // {src, dst}            pt<src, dst>
    {{_f32, _f32},           {bypass(), bypass()}},
    {{_bf16, _bf16 | _f32},  {bypass(), bypass()}},
    {{_any, _any},           {just<f32>(), just<f32>()}},  // fallback
};

static const MappingNotation opNameMappingNotation {
    {ARG_SRC, 0},
    {ARG_DST, 1},
};

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<OpNameAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<OpNameAttrs>> implementations {
        OV_CPU_INSTANCE_X64(
            "op_name_jit_ncsp",
            ExecutorType::Jit,
            OperationType::OpName,
            [](const OpNameConfig& config) -> bool {
                VERIFY(dnnl::impl::cpu::x64::mayiuse(
                    dnnl::impl::cpu::x64::avx2), UNSUPPORTED_ISA);
                // Add precision / attr checks as needed
                return true;
            },
            // createOptimalConfig
            [](const OpNameConfig& config) -> std::optional<OpNameConfig> {
                return createOptimalConfigCommon(config,
                    opNameTypeMapping,
                    LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                    opNameMappingNotation);
            },
            AcceptsAnyShape<OpNameAttrs>,
            CreateDefault<JitOpNameExecutor, OpNameAttrs>{}
            )
        OV_CPU_INSTANCE_COMMON(
            "op_name_ref_ncsp",
            ExecutorType::Reference,
            OperationType::OpName,
            [](const OpNameConfig& config) -> bool { return true; },
            [](const OpNameConfig& config) -> std::optional<OpNameConfig> {
                return createOptimalConfigCommon(config,
                    opNameTypeMapping,
                    LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                    opNameMappingNotation);
            },
            AcceptsAnyShape<OpNameAttrs>,
            CreateDefault<RefOpNameExecutor, OpNameAttrs>{}
            )
    };
    return implementations;
}
// clang-format on

}  // namespace ov::intel_cpu
```

## Implementing concrete executors

Each executor implements the `Executor` base class from `executor.hpp`:

```cpp
class Executor {
public:
    virtual bool update(const MemoryArgs& memory) = 0;   // reshape / re-init
    virtual void execute(const MemoryArgs& memory) = 0;   // run computation
    virtual impl_desc_type implType() const = 0;           // report impl type
    virtual ~Executor() = default;
};
```

**Reference executor** (portable C++):

```cpp
class RefOpNameExecutor : public Executor {
public:
    RefOpNameExecutor(const OpNameAttrs& attrs,
                      const MemoryArgs& memory,
                      const ExecutorContext::CPtr& context);
    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    impl_desc_type implType() const override { return impl_desc_type::ref; }
};
```

**JIT executor** (wraps JIT kernel + CpuParallel):

```cpp
class JitOpNameExecutor : public Executor {
public:
    JitOpNameExecutor(const OpNameAttrs& attrs,
                      const MemoryArgs& memory,
                      const ExecutorContext::CPtr& context);
    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    impl_desc_type implType() const override { return impl_desc_type::jit_avx2; }
private:
    std::shared_ptr<JitKernelBase> m_kernel;
    // ... kernel compile params, context, etc.
};
```

### Wire the node to use ExecutorFactory

In the **node header**, replace direct `execute()` logic with executor fields:

```cpp
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/<op_name>_config.hpp"
#include "nodes/executors/memory_arguments.hpp"

// Inside the class:
private:
    OpNameAttrs m_attrs;
    MemoryArgs m_memory;
    ExecutorFactoryPtr<OpNameAttrs> m_factory;
    ExecutorPtr m_executor = nullptr;
```

In **`initSupportedPrimitiveDescriptors()`** (follows `conv.cpp` pattern):

```cpp
void OpName::initSupportedPrimitiveDescriptors() {
    // 1. Populate m_attrs from the core op
    // m_attrs.epsilon = ...; m_attrs.axis = ...;
    // m_attrs.postOps = getPostOps(fusedWith);

    // 2. Build memory descriptors per port
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    VecMemoryDescs srcDescs;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        srcDescs.push_back(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            getOriginalInputPrecisionAtPort(i), getInputShapeAtPort(i)));
    }
    auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0));

    MemoryDescArgs descs{
        {ARG_SRC, srcDescs[0]},
        {ARG_DST, dstDesc},
    };

    // 3. Create executor factory — this filters & ranks implementations
    auto executionContext = std::make_shared<ExecutorContext>(
        context, getImplPriority());
    m_factory = std::make_shared<ExecutorFactory<OpNameAttrs>>(
        m_attrs, executionContext, descs);

    // 4. Let factory decide optimal memory descriptors
    const auto nodeDescList = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& nodeDescs : nodeDescList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(srcDescs.size());
        // ... populate from nodeDescs (see conv.cpp / fc.cpp pattern) ...
        nodeConfig.outConfs.emplace_back(nodeDescs.at(ARG_DST));
        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}
```

In **`prepareParams()`**:

```cpp
void OpName::prepareParams() {
    m_memory[ARG_SRC] = getSrcMemoryAtPort(0);
    m_memory[ARG_DST] = getDstMemoryAtPort(0);

    if (!m_executor) {
        m_executor = m_factory->make(m_memory);
    }
    m_executor->update(m_memory);
}
```

In **`execute()`**:

```cpp
void OpName::execute([[maybe_unused]] const dnnl::stream& strm) {
    m_executor->execute(m_memory);
}
```

## Executor framework utilities

| Utility | Location | Purpose |
|---------|----------|---------|
| `VERIFY(cond, msg)` | `debug_messages.hpp` | Debug-logged rejection in `supports()` |
| `TypeMapping` / `MappingNotation` | `precision_translation.hpp` | Precision transform rules |
| `createOptimalConfigCommon()` | `implementation_utils.hpp` | Standard config optimisation |
| `AcceptsAnyShape<Attrs>` | `implementation_utils.hpp` | Shape-agnostic stub |
| `HasNoOptimalConfig<Attrs>` | `implementation_utils.hpp` | No config override stub |
| `SupportsAnyConfig<Attrs>` | `implementation_utils.hpp` | Unconditional support stub |
| `CreateDefault<Executor, Attrs>` | `implementation_utils.hpp` | Simple `make_shared` factory |
| `CreateDnnlDefault<Prim, Attrs>` | `implementation_utils.hpp` | `DnnlExecutor` wrapper factory |
| `OV_CPU_INSTANCE_*` macros | `utils/arch_macros.h` | Conditional compilation per arch |

> **Key advantage:** The `ExecutorFactory` automatically handles runtime
> selection between JIT / oneDNN / ACL / Reference implementations based on
> hardware capabilities, precision matching, and memory format preferences.
> When multiple implementations match, `VariableExecutor` handles fallback.

## CpuParallel multithreading

The simplest and most common optimization. Use `CpuParallel` (available via
`context->getCpuParallel()`) to parallelise the execution loop inside any
executor — Reference, JIT, or oneDNN.

**File:** `src/plugins/intel_cpu/src/cpu_parallel.hpp`

**Available methods:**

| Method | Signature | Use Case |
|--------|-----------|----------|
| `parallel_for` | `(D0, func(i))` | 1D iteration |
| `parallel_for2d` | `(D0, D1, func(i, j))` | 2D iteration |
| `parallel_for3d` | `(D0, D1, D2, func(i, j, k))` | 3D iteration (batch × spatial) |
| `parallel_for4d` | `(D0, D1, D2, D3, func(...))` | 4D (batch × channels × H × W) |
| `parallel_for5d` | `(D0, D1, D2, D3, D4, func(...))` | 5D tensors |
| `parallel_for6d` | `(D0, ..., D5, func(...))` | 6D tensors |
| `parallel_sum` | `(D0, init, func(i))` | Parallel reduction |
| `parallel_sum2d` | `(D0, D1, init, func(i, j))` | 2D parallel reduction |
| `parallel_sum3d` | `(D0, D1, D2, init, func(i, j, k))` | 3D parallel reduction |
| `parallel_simple` | `(nthr, func(ithr, nthr))` | Manual thread partitioning |

**Example — element-wise op with CpuParallel inside a Reference executor:**

```cpp
void RefOpNameExecutor::execute(const MemoryArgs& memory) {
    auto* src = memory.at(ARG_SRC)->getDataAs<const float>();
    auto* dst = memory.at(ARG_DST)->getDataAs<float>();
    auto total = memory.at(ARG_SRC)->getShape().getElementsCount();

    m_context->getCpuParallel()->parallel_for(total, [&](size_t i) {
        dst[i] = /* element-wise computation on */ src[i];
    });
}
```

**Example — batch-parallelised reduction:**

```cpp
const auto& shape = memory.at(ARG_SRC)->getStaticDims();
auto batch = shape_size(shape) / shape.back();
auto inner = shape.back();

m_context->getCpuParallel()->parallel_for(batch, [&](size_t b) {
    auto* row_src = src + b * inner;
    auto* row_dst = dst + b * inner;
    // Per-row reduction + normalization
    float sum = 0.0f;
    for (size_t i = 0; i < inner; i++) {
        sum += row_src[i] * row_src[i];
    }
    float scale = 1.0f / std::sqrt(sum / inner + eps);
    for (size_t i = 0; i < inner; i++) {
        row_dst[i] = row_src[i] * scale;
    }
});
```

## Writing a JIT kernel and JIT executor

JIT kernels are **not** used directly from the node class. Instead, they are
wrapped inside an executor class that implements the `Executor` interface and
is registered as an `ExecutorType::Jit` entry in `<op_name>_implementations.cpp`.

The ISA dispatch (`mayiuse(avx512_core)` vs `mayiuse(avx2)`) happens in the
`supports()` predicate of the `ExecutorImplementation`, **not** in the node.

**Study** `eltwise_implementations.cpp` — it shows Jit executors
(`eltwise_jit_ncsp`, `eltwise_jit_nspc`) alongside Reference executors, all
as `ExecutorImplementation<EltwiseAttrs>` entries.

For the JIT emitter internals (xbyak, register allocation, debugging), see the
[emitters README](../../src/emitters/README.md); for the SIMD abstraction layer
(`vec<T, isa>` and writing portable SIMD inside a kernel), see the
[SIMD README](../../src/nodes/kernels/simd/README.md).

### Step 1: Write the JIT kernel

**Directory:** `src/plugins/intel_cpu/src/nodes/kernels/x64/`

```cpp
// <op_name>_kernel.hpp
#pragma once

#include "nodes/kernels/x64/jit_kernel_base.hpp"

namespace ov::intel_cpu::kernel {

struct jit_<op_name>_compile_params {
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    size_t data_size;
    // Op-specific compile-time parameters
};

struct jit_<op_name>_call_args {
    const uint8_t* src;
    uint8_t* dst;
    // Op-specific runtime arguments
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class jit_<op_name>_kernel : public JitKernelBase {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_<op_name>_kernel)

    explicit jit_<op_name>_kernel(const jit_<op_name>_compile_params& jcp);
    void generate() override;

private:
    jit_<op_name>_compile_params m_jcp;
    // Register assignments, helper methods
};

}  // namespace ov::intel_cpu::kernel
```

### Step 2: Wrap the kernel in a JIT executor class

Place in `src/plugins/intel_cpu/src/nodes/executors/jit/<op_name>_jit.hpp`:

```cpp
#pragma once

#include "nodes/executors/executor.hpp"
#include "nodes/executors/<op_name>_config.hpp"
#include "nodes/executors/executor_context.hpp"

namespace ov::intel_cpu {

class JitOpNameExecutor : public Executor {
public:
    JitOpNameExecutor(const OpNameAttrs& attrs,
                      const MemoryArgs& memory,
                      const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    impl_desc_type implType() const override { return impl_desc_type::jit_avx2; }

private:
    OpNameAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    std::shared_ptr<kernel::JitKernelBase> m_kernel;
    // ... cached shape info, working buffers
};
```

The constructor creates / compiles the JIT kernel. `update()` re-compiles if
shapes changed. `execute()` sets up call args and invokes the kernel inside
a `CpuParallel` loop over the outer dimension.

### Step 3: Register in `<op_name>_implementations.cpp`

```cpp
OV_CPU_INSTANCE_X64(
    "op_name_jit_ncsp",
    ExecutorType::Jit,
    OperationType::OpName,
    [](const OpNameConfig& config) -> bool {
        VERIFY(dnnl::impl::cpu::x64::mayiuse(
            dnnl::impl::cpu::x64::avx2), UNSUPPORTED_ISA);
        return true;
    },
    [](const OpNameConfig& config) -> std::optional<OpNameConfig> {
        return createOptimalConfigCommon(config,
            opNameTypeMapping,
            LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
            opNameMappingNotation);
    },
    AcceptsAnyShape<OpNameAttrs>,
    CreateDefault<JitOpNameExecutor, OpNameAttrs>{}
    )
```

For an AVX-512 variant, add a **second** entry before the AVX2 one (higher
priority first) with `mayiuse(avx512_core)` in the `supports` predicate and
a correspondingly wider implementation.

**ISA-specific considerations:**

| ISA | Vector Width | Key Features |
|-----|-------------|--------------|
| AVX2 | 256-bit (8×f32) | `ymm` registers, FMA, most common JIT baseline |
| AVX-512 (Core) | 512-bit (16×f32) | `zmm` registers, mask registers, wider SIMD |
| AVX-512 BF16 | 512-bit | `avx512_core_bf16` — required for BF16 compute |
| AVX-512 VNNI | 512-bit | `vpdpbusd` for INT8 dot products |
| AMX | Tile-based | `tdpbf16ps` / `tdpbssd` for matrix ops |

> **Note:** SSE 4.2 is no longer a target ISA. The minimal baseline is portable
> C++ code. JIT kernels start from AVX2.

**Reference JIT executor implementations to study:**

| Op | Executor | Key Pattern |
|----|----------|-------------|
| Eltwise | `EltwiseStatefulExecutor` (Jit + Ref) | Element-wise, multiple layouts |
| RMSNorm | `nodes/kernels/x64/rms_kernel.hpp` | Reduction + normalize |
| RoPE | `nodes/kernels/x64/rope_kernel.hpp` | Element-wise with indexing |

## oneDNN executor implementations

For ops that map to oneDNN primitives (convolutions, matmul, pooling, softmax,
layer normalization, etc.), create an `ExecutorType::Dnnl` implementation using
the `DnnlExecutor` wrapper.

**Study** `convolution_implementations.cpp` — it registers multiple oneDNN
variants differentiated by memory layout (nspc, ncsp, nCsp16c, nCsp8c), each
as a separate `ExecutorImplementation<ConvAttrs>`.

### DnnlExecutor pattern

Use the `CreateDnnlDefault<Primitive, Attrs>` helper to create a standard
`DnnlExecutor` that wraps a oneDNN primitive:

```cpp
OV_CPU_INSTANCE_DNNL_X64(
    "op_name_dnnl_nspc",
    ExecutorType::Dnnl,
    OperationType::OpName,
    [](const OpNameConfig& config, const MemoryFormatFilter& filter) -> bool {
        VERIFY(filter.isSuitableMemoryFormatFilter(
            MemoryFormatFilter::nspc), UNSUPPORTED_MEMORY_FORMAT);
        return true;
    },
    [](const OpNameConfig& config) -> std::optional<OpNameConfig> {
        return createOptimalConfigCommon(config,
            opNameDnnlTypeMapping,
            LayoutConfig{LayoutType::nspc, LayoutType::nspc},
            opNameMappingNotation);
    },
    AcceptsAnyShape<OpNameAttrs>,
    CreateDnnlDefault<DnnlOpNamePrimitive, OpNameAttrs>{}
    )
```

The `DnnlExecutor<Primitive, Attrs, ShapeAgnosticData>` template:
- Calls `Primitive::createDescriptors()` to build oneDNN primitive descriptors
- Manages primitive caching via `ExecutorContext`
- Implements `update()` / `execute()` around the oneDNN primitive execution

**Note:** `CreateDnnlDefault` accepts optional flags: `{cacheWeights, fc3Das2D}`.

### `MemoryFormatFilter`

oneDNN implementations typically require specific memory layouts. Use the
extended `supports` predicate form that takes `MemoryFormatFilter`:

```cpp
[](const OpNameConfig& config, const MemoryFormatFilter& filter) -> bool {
    VERIFY(filter.isSuitableMemoryFormatFilter(
        MemoryFormatFilter::nspc), UNSUPPORTED_MEMORY_FORMAT);
    // additional ISA / precision checks
    return true;
}
```

For the convolution-specific post-op and zero-point mechanisms, see
[Convolution post ops](../convolution_post_ops.md).

## Eltwise JIT emitters

For ops routed through the `Eltwise` node (see
[the fast path](./implementing_a_node.md#fast-path-routing-a-unary-elementwise-op-through-eltwise)),
JIT execution uses **eltwise emitter classes** rather than the executor
framework. Each ISA has its own emitter base class. The emitter lifecycle
(`emit_impl`, `register_table_entries`, register allocation) is described in the
[emitters README](../../src/emitters/README.md).

### Emitter class pattern

```cpp
// Header (jit_eltwise_emitters.hpp)
class jit_<op_name>_emitter : public jit_emitter {
public:
    jit_<op_name>_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                          dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                          ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_count() const override { return 1; }

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs,
                   const std::vector<size_t>& out_vec_idxs) const override;
    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs,
                  const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
    size_t aux_vecs_count() const override;  // return count of needed aux vmm registers
};
```

### Coefficient tables

Use `push_arg_entry_of("name", hex_value, true)` in `register_table_entries()`:

```cpp
void jit_<op_name>_emitter::register_table_entries() {
    // Float constants as hex IEEE 754
    push_arg_entry_of("coeff_a", 0x3f800000, true);  // 1.0f
    push_arg_entry_of("abs_mask", 0x7fffffff, true);  // strip sign bit
    push_arg_entry_of("pos_inf",  0x7f800000, true);  // +inf
    push_arg_entry_of("qnan",     0x7fc00000, true);  // quiet NaN
}
// Access in emit_isa: table_val("coeff_a")
```

## Functional verification

After optimization, verify correctness before proceeding to full testing:

```bash
# Quick functional check
cd build
cmake --build . --target ov_cpu_func_tests -j$(nproc)
./bin/intel64/Release/ov_cpu_func_tests --gtest_filter=*<OpName>*
```

Then proceed to [Testing a CPU Op](./testing_a_node.md).
