// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu {

struct jit_args_dft {
    const float* src;
    float* dst;
    const float* twiddles;

    size_t work_amount;
    size_t index;
};

struct jit_args_fft {
    const float* src;
    float* dst;
    const float* twiddles;

    size_t num_blocks;
    size_t work_amount;
    size_t n_complex;
};

struct jit_uni_dft_kernel {
    void (*ker_)(const jit_args_dft*){nullptr};

    void operator()(const jit_args_dft* args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_dft_kernel() = default;
    virtual ~jit_uni_dft_kernel() = default;

    virtual void create_ker() = 0;
};

struct jit_uni_fft_kernel {
    void (*ker_)(const jit_args_fft*){nullptr};

    void operator()(const jit_args_fft* args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_fft_kernel() = default;
    virtual ~jit_uni_fft_kernel() = default;

    virtual void create_ker() = 0;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_uni_dft_kernel_f32 : public jit_uni_dft_kernel, public dnnl::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dft_kernel_f32)

    jit_uni_dft_kernel_f32();

    void create_ker() override;
    void generate() override;

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;
    size_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_twiddles = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_index = r12;
    Xbyak::Reg64 reg_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    Vmm vmm_data = Vmm(0);
    Vmm vmm_twiddles = Vmm(1);
    Vmm vmm_sum = Vmm(2);
    Vmm vmm_sum_2 = vmm_data;
    Vmm vmm_data_cache = Vmm(3);
    Vmm vmm_twiddles_cache = Vmm(4);

    Xbyak::Xmm xmm_data = Xbyak::Xmm(0);
    Xbyak::Xmm xmm_twiddles = Xbyak::Xmm(1);
    Xbyak::Xmm xmm_sum = Xbyak::Xmm(2);
    Xbyak::Xmm xmm_sum_2 = xmm_data;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_uni_fft_kernel_f32 : public jit_uni_fft_kernel, public dnnl::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_fft_kernel_f32)

    jit_uni_fft_kernel_f32();

    void create_ker() override;
    void generate() override;

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;
    const size_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_even_in_diff = rax;
    Xbyak::Reg64 reg_even_out_diff = rbx;

    Xbyak::Reg64 reg_src = r9;
    Xbyak::Reg64 reg_dst = r10;
    Xbyak::Reg64 reg_num_blocks = r11;
    Xbyak::Reg64 reg_work_amount = r12;
    Xbyak::Reg64 aux_reg_work_amount = r13;
    Xbyak::Reg64 reg_twiddles_addr = r14;
    Xbyak::Reg64 reg_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    Vmm vmm_data_odd_1 = Vmm(0);
    Vmm vmm_data_odd_2 = Vmm(1);
    Vmm vmm_twiddle_real = Vmm(2);
    Vmm vmm_twiddle_imag = Vmm(3);
    Vmm vmm_data_even = Vmm(4);

    Vmm vmm_data_result = vmm_data_odd_2;

    template <typename T>
    void loop_process(int step);

    void move_data(const Xbyak::Address& addr, const Xbyak::Xmm& x, int count);
    void move_data(const Xbyak::Xmm& x, const Xbyak::Address& addr, int count);
};

}  // namespace ov::intel_cpu
