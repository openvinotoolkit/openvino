# Executors, Kernels and Optimization

This is the optimization phase of adding an operation to the Intel CPU plugin. It
covers what you create for *your* op when it needs the executor framework, and
when that phase can be skipped. The framework itself — architecture,
`ExecutorImplementation`, `getImplementations` / `OV_CPU_INSTANCE_*`, concrete
Ref/JIT/oneDNN executors, the utilities table, `CpuParallel`, JIT kernels, and
oneDNN `DnnlExecutor`/`MemoryFormatFilter` — is documented once in the
[executor framework reference](../../src/nodes/executors/README.md); this page
links into it rather than repeating it.

It assumes the node compiles and runs via a reference path (see
[Implementing a CPU Node](./implementing_a_node.md)) and that the ISA targets and
strategy were chosen during
[analysis](./choosing_a_strategy.md#the-strategy-decision-matrix).

## Contents

- [When to optimize (and when to skip)](#when-to-optimize-and-when-to-skip)
- [What you create for your op](#what-you-create-for-your-op)
- [Eltwise JIT emitters](#eltwise-jit-emitters)

## When to optimize (and when to skip)

Optimization is **mandatory** for performance-critical ops (element-wise,
reductions, attention, normalization). It may be **skipped** for simple ops where
the reference implementation is sufficient.

Skip criteria:
- Op is rarely used in production models.
- Reference implementation already meets performance requirements.
- Op is pure data movement (handled efficiently by `Reorder` or `Reshape`).

## What you create for your op

For an executor-based op you add, alongside the node:

1. **`<op_name>_config.hpp`** — the `OpNameAttrs` struct (the op-specific
   attributes every implementation needs) and the `OpNameConfig` alias. Structure
   described in
   [ExecutorImplementation / getImplementations](../../src/nodes/executors/README.md#getimplementations-and-the-ov_cpu_instance-macros).
2. **`<op_name>_implementations.cpp`** — the `getImplementations<OpNameAttrs>()`
   specialisation listing your implementations in priority order, each wrapped in
   the appropriate `OV_CPU_INSTANCE_*` macro. See
   [getImplementations and the OV_CPU_INSTANCE macros](../../src/nodes/executors/README.md#getimplementations-and-the-ov_cpu_instance-macros).
3. **The concrete executor classes** (Reference, JIT, oneDNN) implementing the
   `Executor` interface. See
   [Concrete executors](../../src/nodes/executors/README.md#concrete-executors).
4. **Node wiring** — hold an `ExecutorFactory` and drive it from
   `initSupportedPrimitiveDescriptors` / `prepareParams` / `execute`. See
   [Wiring a node to an ExecutorFactory](../../src/nodes/executors/README.md#wiring-a-node-to-an-executorfactory).

The optimization technique you reach for depends on the op:

- **Multithreading** — almost always, parallelise the loop with `CpuParallel`. See
  [CpuParallel multithreading](../../src/nodes/executors/README.md#cpuparallel-multithreading).
- **JIT kernel** — for SIMD-friendly ops, add an `ExecutorType::Jit` implementation
  wrapping a kernel. See
  [Writing a JIT kernel](../../src/nodes/executors/README.md#writing-a-jit-kernel)
  and the ISA-targets table there.
- **oneDNN primitive** — for ops that map to a oneDNN primitive (conv, matmul,
  pooling, …), add an `ExecutorType::Dnnl` implementation. See
  [oneDNN executors](../../src/nodes/executors/README.md#onednn-executors).

Study `eltwise_implementations.cpp` (simplest), `fullyconnected_implementations.cpp`
(full-featured), and `convolution_implementations.cpp` (layout-aware) as canonical
examples.

## Eltwise JIT emitters

If your op took the
[Eltwise fast path](./implementing_a_node.md#fast-path-routing-a-unary-elementwise-op-through-eltwise),
its JIT execution uses **eltwise emitter classes** rather than the executor
framework. Each ISA has its own emitter base class; the lifecycle (`emit_impl`,
`register_table_entries`, register allocation) is described in the
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
    size_t aux_vecs_count() const override;  // count of needed aux vmm registers
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
