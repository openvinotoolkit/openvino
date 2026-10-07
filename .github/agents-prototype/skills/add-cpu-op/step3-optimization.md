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

1. Read [`executors/README.md`](../../../../src/plugins/intel_cpu/src/nodes/executors/README.md) —
   it covers `CpuParallel` multithreading (the method table), writing a JIT
   kernel and registering it via `OV_CPU_INSTANCE_*` (the ISA table), and
   registering oneDNN executor implementations via `CreateDnnlDefault`.
2. Parallelise the executor's inner loop with `CpuParallel`, and add JIT or
   oneDNN executor implementations as needed, following that guide's patterns.

## JIT Eltwise Emitters

For ops routed through the `Eltwise` node (see "Fast Path" in [**cpu_op_implementation**](step2-implementation.md)), JIT execution uses **eltwise emitter classes** rather than the executor framework. Each ISA has its own emitter base class.

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
