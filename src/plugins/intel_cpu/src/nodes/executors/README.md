# CPU Plugin Executor Framework

The executor framework is the standard architecture for any CPU node more complex
than portable C++. A node holds an attributes struct and an `ExecutorFactory`; the
factory filters and ranks a list of `ExecutorImplementation` entries by ISA,
precision, and memory layout, then produces an `ExecutorPtr` that the node drives.
JIT kernels, oneDNN primitives, ACL/MLAS/KleidiAI backends, and a portable
reference all coexist as separate implementations of the same op.

This is the reference for the framework itself. For the workflow of adding a new
op (what files *you* create for *your* op) see
[docs/op_development](../../../docs/op_development/README.md); for the node side
(lifecycle, registration) see [../README.md](../README.md).

## Contents

- [Architecture](#architecture)
- [ExecutorImplementation](#executorimplementation)
- [getImplementations and the OV_CPU_INSTANCE macros](#getimplementations-and-the-ov_cpu_instance-macros)
- [Concrete executors](#concrete-executors)
- [Wiring a node to an ExecutorFactory](#wiring-a-node-to-an-executorfactory)
- [Framework utilities](#framework-utilities)
- [CpuParallel multithreading](#cpuparallel-multithreading)
- [oneDNN executors](#onednn-executors)

Study these canonical implementations in order: `eltwise_implementations.cpp`
(simplest — Jit + ACL + Reference), `fullyconnected_implementations.cpp`
(full-featured — MLAS, 1x1conv, ACL, KleidiAI, oneDNN),
`convolution_implementations.cpp` (layout-aware with `MemoryFormatFilter`).

## Architecture

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

The `ExecutorFactory` handles runtime selection between JIT / oneDNN / ACL /
Reference based on hardware, precision matching, and memory-format preferences.
When several implementations match, `VariableExecutor` handles fallback.

## ExecutorImplementation

Each `ExecutorImplementation<Attrs>` has these fields:

1. **name** — unique human-readable string (e.g. `"op_name_jit_ncsp"`).
2. **ExecutorType** — `Jit`, `Dnnl`, `Acl`, `Reference`, etc.
3. **OperationType** — `OperationType::OpName`.
4. **supports** — lambda returning `bool`. Use the `VERIFY(condition, MESSAGE)`
   macro for debuggable rejection. Can take `(const Config&)` or
   `(const Config&, const MemoryFormatFilter&)`.
5. **createOptimalConfig** — lambda returning `std::optional<Config>`. Return `{}`
   (nullopt) if the current memory config is already acceptable; otherwise use
   `createOptimalConfigCommon()` with a `TypeMapping` and `LayoutConfig`.
6. **acceptsShape** — lambda `(const Attrs&, const MemoryArgs&) -> bool`. Pass
   `AcceptsAnyShape<Attrs>` for shape-agnostic implementations.
7. **create** — lambda returning `ExecutorPtr`. Use `CreateDefault<Executor, Attrs>{}`
   for simple cases, or `CreateDnnlDefault<Primitive, Attrs>{}` / a custom lambda.

## getImplementations and the OV_CPU_INSTANCE macros

A `getImplementations<OpNameAttrs>()` specialisation returns the implementation
list in **priority order** (highest first). Each entry is wrapped in an
`OV_CPU_INSTANCE_*` macro (defined in [../../utils/arch_macros.h](../../utils/arch_macros.h))
that conditionally compiles it for the target architecture, so code for
unavailable platforms is compiled away (see
[selective_build.md](../../../docs/selective_build.md#in-code-conditional-compilation)).

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

```cpp
using LayoutConfig = std::vector<LayoutType>;

// TypeMapping defines precision transformations per implementation.
// Each row: {input precisions, output precisions} → {transform functions}.
// Helpers: bypass() = keep, just<T>() = force to T, use<N>() = match arg N.
static const TypeMapping opNameTypeMapping {
    {{_f32, _f32},           {bypass(), bypass()}},
    {{_bf16, _bf16 | _f32},  {bypass(), bypass()}},
    {{_any, _any},           {just<f32>(), just<f32>()}},  // fallback
};

static const MappingNotation opNameMappingNotation { {ARG_SRC, 0}, {ARG_DST, 1} };

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
                return true;
            },
            [](const OpNameConfig& config) -> std::optional<OpNameConfig> {
                return createOptimalConfigCommon(config, opNameTypeMapping,
                    LayoutConfig{LayoutType::ncsp, LayoutType::ncsp}, opNameMappingNotation);
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
                return createOptimalConfigCommon(config, opNameTypeMapping,
                    LayoutConfig{LayoutType::ncsp, LayoutType::ncsp}, opNameMappingNotation);
            },
            AcceptsAnyShape<OpNameAttrs>,
            CreateDefault<RefOpNameExecutor, OpNameAttrs>{}
            )
    };
    return implementations;
}
// clang-format on
```

For an AVX-512 variant, add a **second** `OV_CPU_INSTANCE_X64` entry **before** the
AVX2 one (higher priority first) with `mayiuse(avx512_core)` in `supports` and a
correspondingly wider implementation.

The attributes struct and `Config` alias live in `<op_name>_config.hpp`:

```cpp
struct OpNameAttrs {
    // Op-specific attributes needed by all implementations.
    PostOps postOps;  // include if the op supports post-op fusion
};
using OpNameConfig = executor::Config<OpNameAttrs>;
```

## Concrete executors

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

A reference executor returns `impl_desc_type::ref`; a JIT executor wraps a kernel
(see [JIT kernels](#cpuparallel-multithreading) below and the
[emitters README](../../emitters/README.md)) and returns e.g.
`impl_desc_type::jit_avx2`. ISA dispatch (`mayiuse(avx512_core)` vs `mayiuse(avx2)`)
happens in the implementation's `supports()` predicate, **not** in the node or the
executor.

```cpp
class JitOpNameExecutor : public Executor {
public:
    JitOpNameExecutor(const OpNameAttrs& attrs, const MemoryArgs& memory,
                      const ExecutorContext::CPtr& context);
    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    impl_desc_type implType() const override { return impl_desc_type::jit_avx2; }
private:
    OpNameAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    std::shared_ptr<kernel::JitKernelBase> m_kernel;
};
```

The constructor compiles the JIT kernel; `update()` recompiles on shape change;
`execute()` sets up call args and invokes the kernel inside a `CpuParallel` loop.

## Wiring a node to an ExecutorFactory

In the node header, hold the attrs, memory, factory, and executor:

```cpp
private:
    OpNameAttrs m_attrs;
    MemoryArgs m_memory;
    ExecutorFactoryPtr<OpNameAttrs> m_factory;
    ExecutorPtr m_executor = nullptr;
```

In `initSupportedPrimitiveDescriptors()` (follows `conv.cpp`): populate `m_attrs`,
build per-port `MemoryDescArgs`, create the factory, and let it pick the optimal
memory descriptors:

```cpp
auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority());
m_factory = std::make_shared<ExecutorFactory<OpNameAttrs>>(m_attrs, executionContext, descs);
const auto nodeDescList = m_factory->getProperMemoryDescriptors(descs);
// emplace one supportedPrimitiveDescriptor per returned descriptor set
```

In `prepareParams()`: refresh `m_memory`, create the executor once, then `update`:

```cpp
m_memory[ARG_SRC] = getSrcMemoryAtPort(0);
m_memory[ARG_DST] = getDstMemoryAtPort(0);
if (!m_executor) {
    m_executor = m_factory->make(m_memory);
}
m_executor->update(m_memory);
```

In `execute()`: `m_executor->execute(m_memory);`

## Framework utilities

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
| `OV_CPU_INSTANCE_*` macros | `../../utils/arch_macros.h` | Conditional compilation per arch |

## CpuParallel multithreading

`CpuParallel` (via `context->getCpuParallel()`, defined in
[../../cpu_parallel.hpp](../../cpu_parallel.hpp)) parallelises the execution loop
inside any executor — Reference, JIT, or oneDNN.

| Method | Signature | Use Case |
|--------|-----------|----------|
| `parallel_for` | `(D0, func(i))` | 1D iteration |
| `parallel_for2d` | `(D0, D1, func(i, j))` | 2D iteration |
| `parallel_for3d` | `(D0, D1, D2, func(i, j, k))` | 3D (batch × spatial) |
| `parallel_for4d` | `(D0, D1, D2, D3, func(...))` | 4D (batch × channels × H × W) |
| `parallel_for5d` | `(D0, D1, D2, D3, D4, func(...))` | 5D tensors |
| `parallel_for6d` | `(D0, ..., D5, func(...))` | 6D tensors |
| `parallel_sum` | `(D0, init, func(i))` | Parallel reduction |
| `parallel_sum2d` | `(D0, D1, init, func(i, j))` | 2D parallel reduction |
| `parallel_sum3d` | `(D0, D1, D2, init, func(i, j, k))` | 3D parallel reduction |
| `parallel_simple` | `(nthr, func(ithr, nthr))` | Manual thread partitioning |

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

### Writing a JIT kernel

A JIT kernel lives under [kernels/x64/](../kernels/x64/) with compile-params and
call-args structs and a `jit_<op_name>_kernel<isa>` template; the executor above
wraps it. For emitter internals (xbyak, register allocation, debugging) see the
[emitters README](../../emitters/README.md); for the portable SIMD abstraction
(`vec<T, isa>`) see the [SIMD README](../kernels/simd/README.md).

| ISA | Vector Width | Key Features |
|-----|-------------|--------------|
| AVX2 | 256-bit (8×f32) | `ymm` registers, FMA, most common JIT baseline |
| AVX-512 (Core) | 512-bit (16×f32) | `zmm` registers, mask registers, wider SIMD |
| AVX-512 BF16 | 512-bit | `avx512_core_bf16` — required for BF16 compute |
| AVX-512 VNNI | 512-bit | `vpdpbusd` for INT8 dot products |
| AMX | Tile-based | `tdpbf16ps` / `tdpbssd` for matrix ops |

> SSE 4.2 is no longer a target ISA. The minimal baseline is portable C++ code;
> JIT kernels start from AVX2.

Reference JIT executors to study: Eltwise (`EltwiseStatefulExecutor`, multiple
layouts), RMSNorm (`kernels/x64/rms_kernel.hpp`, reduction + normalize), RoPE
(`kernels/x64/rope_kernel.hpp`, element-wise with indexing).

## oneDNN executors

For ops that map to oneDNN primitives (convolution, matmul, pooling, softmax,
layer norm, …), register an `ExecutorType::Dnnl` implementation built with
`CreateDnnlDefault<Primitive, Attrs>`. `convolution_implementations.cpp` registers
several oneDNN variants differentiated by memory layout (nspc, ncsp, nCsp16c,
nCsp8c).

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
        return createOptimalConfigCommon(config, opNameDnnlTypeMapping,
            LayoutConfig{LayoutType::nspc, LayoutType::nspc}, opNameMappingNotation);
    },
    AcceptsAnyShape<OpNameAttrs>,
    CreateDnnlDefault<DnnlOpNamePrimitive, OpNameAttrs>{}
    )
```

The `DnnlExecutor<Primitive, Attrs, ShapeAgnosticData>` template calls
`Primitive::createDescriptors()`, manages primitive caching via `ExecutorContext`,
and implements `update()`/`execute()` around the oneDNN primitive.
`CreateDnnlDefault` accepts optional flags `{cacheWeights, fc3Das2D}`.
`MemoryFormatFilter` (in the extended `supports` predicate) expresses required
layouts. For convolution-specific post-op and zero-point mechanisms see
[convolution_post_ops.md](../../../docs/convolution_post_ops.md).
