# Skill: CPU Op Optimization

> Agent: `cpu_agent` — Step 3 of 4

## Prerequisites

- Completed **cpu_op_implementation** skill — node class compiles and executes
  via reference implementation.
- ISA targets and implementation strategy determined in **cpu_op_analysis**.

## When to Apply

This step is **mandatory** for performance-critical ops (element-wise, reductions,
attention, normalization). It may be **skipped** for simple ops where the reference
implementation is sufficient (mark the Step 3 checkpoint or step-local status as
`skipped`; the final job `status` should still be `success` if implementation and
tests pass).

Skip criteria:
- Op is rarely used in production models.
- Reference implementation already meets performance requirements.
- Op is pure data movement (handled efficiently by `Reorder` or `Reshape`).

## Optimization Techniques

All optimization techniques described below are used **inside executor
implementations**. The executor framework handles ISA dispatch, fallback
ordering, and precision matching — see `cpu_op_implementation` skill for
the full wiring pattern. This skill focuses on the code that goes **inside**
each executor class.

### CpuParallel Multithreading

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

### Creating JIT Executor Implementations

JIT kernels are **not** used directly from the node class. Instead, they are
wrapped inside an executor class that implements the `Executor` interface and
is registered as an `ExecutorType::Jit` entry in `<op_name>_implementations.cpp`.

The ISA dispatch (`mayiuse(avx512_core)` vs `mayiuse(avx2)`) happens in the
`supports()` predicate of the `ExecutorImplementation`, **not** in the node.

**Study** `eltwise_implementations.cpp` — it shows Jit executors
(`eltwise_jit_ncsp`, `eltwise_jit_nspc`) alongside Reference executors, all
as `ExecutorImplementation<EltwiseAttrs>` entries.

#### Step 1: Write the JIT Kernel

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

#### Step 2: Wrap the Kernel in a JIT Executor Class

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

#### Step 3: Register in `<op_name>_implementations.cpp`

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

### Creating oneDNN Executor Implementations

For ops that map to oneDNN primitives (convolutions, matmul, pooling, softmax,
layer normalization, etc.), create an `ExecutorType::Dnnl` implementation using
the `DnnlExecutor` wrapper.

**Study** `convolution_implementations.cpp` — it registers multiple oneDNN
variants differentiated by memory layout (nspc, ncsp, nCsp16c, nCsp8c), each
as a separate `ExecutorImplementation<ConvAttrs>`.

#### DnnlExecutor Pattern

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

#### `MemoryFormatFilter`

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

## JIT Eltwise Emitters

For ops routed through the `Eltwise` node (see "Fast Path" in [**cpu_op_implementation**](skills/add-cpu-op/step2-implementation.md)), JIT execution uses **eltwise emitter classes** rather than the executor framework. Each ISA has its own emitter base class.

### Emitter Class Pattern

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

### Coefficient Tables

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

## Functional Verification

After optimization, verify correctness before proceeding to full testing:

```bash
# Quick functional check
cd build
cmake --build . --target ov_cpu_func_tests -j$(nproc)
./bin/intel64/Release/ov_cpu_func_tests --gtest_filter=*<OpName>*
```

## Output

- Optimized implementation with ISA-specific paths.
- JIT kernels created (if applicable).
- `CpuParallel` integration for multi-threaded execution.
- Functional verification passes.
- Proceed to **cpu_op_testing** skill.
