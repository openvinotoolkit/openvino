// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "jit_kernel_base.hpp"

namespace ov::intel_cpu::kernel {

struct jit_selective_ssm_compile_params {
    size_t state_size = 0;
};

struct jit_selective_ssm_call_args {
    const float* state_src = nullptr;
    float* state_dst = nullptr;
    const float* B = nullptr;
    const float* C = nullptr;
    float decay = 0.F;
    float delta = 0.F;
    const float* x = nullptr;
    size_t p_count = 0;
    float* output = nullptr;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_selective_ssm_kernel : public JitKernel<jit_selective_ssm_compile_params, jit_selective_ssm_call_args> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_selective_ssm_kernel)

    explicit jit_selective_ssm_kernel(const jit_selective_ssm_compile_params& jcp)
        : JitKernel<jit_selective_ssm_compile_params, jit_selective_ssm_call_args>(jit_name(), jcp, isa) {}

private:
    using Vmm = std::conditional_t<isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>;

    static constexpr size_t vec_size = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen / sizeof(float);
    static constexpr size_t vec_bytes = vec_size * sizeof(float);

    const Xbyak::Reg64 reg_args = rbx;
    const Xbyak::Reg64 reg_state_src = r8;
    const Xbyak::Reg64 reg_state_dst = r9;
    const Xbyak::Reg64 reg_B = r10;
    const Xbyak::Reg64 reg_C = r11;
    const Xbyak::Reg64 reg_x = r12;
    const Xbyak::Reg64 reg_p_count = r13;
    const Xbyak::Reg64 reg_output = rax;

    const Vmm v_decay = Vmm(0);
    const Vmm v_input_scale = Vmm(1);
    const Vmm v_delta = Vmm(11);
    const Xbyak::Xmm x_acc = Xbyak::Xmm(2);
    const Vmm v_state = Vmm(3);
    const Vmm v_B = Vmm(4);
    const Vmm v_C = Vmm(5);

    const Xbyak::Xmm x_state = Xbyak::Xmm(6);
    const Xbyak::Xmm x_B = Xbyak::Xmm(7);
    const Xbyak::Xmm x_C = Xbyak::Xmm(8);
    const Xbyak::Xmm x_sum = Xbyak::Xmm(9);
    const Xbyak::Xmm x_tmp = Xbyak::Xmm(10);

    void generate() override;
};

std::shared_ptr<JitKernelBase> create_selective_ssm_jit_kernel(size_t state_size, bool prefer_avx512);

}  // namespace ov::intel_cpu::kernel
