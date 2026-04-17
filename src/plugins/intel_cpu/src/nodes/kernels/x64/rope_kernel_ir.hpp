// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// IR-mode RoPE kernel using jit_kernel DSL with register allocation.
// Drop-in replacement for jit_rotary_kernel — same call args, same semantics.
// Selected at runtime via OV_JIT_IR_ROPE=1 env var.

#pragma once

#include "jit_kernel.hpp"
#include "jit_kernel_base.hpp"
#include "rope_kernel.hpp"  // jit_rotary_compile_params, jit_rotary_call_args

#if defined(OPENVINO_ARCH_X86_64)

namespace ov::intel_cpu::kernel {

struct jit_rotary_kernel_ir : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rotary_kernel_ir)

    using Params = jit_rotary_call_args;
    using function_t = void (*)(const Params*);

    explicit jit_rotary_kernel_ir(const jit_rotary_compile_params& jcp);

    void init();

    void operator()(const Params* args) const {
        _fn(args);
    }

    // Adapter so execJitKernel (which calls (*ker)(&args)) works via JitKernelBase.
    void operator()(const void* args) const {
        _fn(static_cast<const Params*>(args));
    }

private:
    void generate() override;

    template <typename T, size_t N>
    void rotary_half_ir();

    template <typename T, size_t N>
    void rotary_interleave_ir();

    jit_rotary_compile_params m_jcp;
    function_t _fn = nullptr;
};

}  // namespace ov::intel_cpu::kernel

#endif  // OPENVINO_ARCH_X86_64
