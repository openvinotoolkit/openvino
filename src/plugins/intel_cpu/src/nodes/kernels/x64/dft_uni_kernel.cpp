// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dft_uni_kernel.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;

#define GET_OFF_DFT(field) offsetof(jit_args_dft, field)
#define GET_OFF_FFT(field) offsetof(jit_args_fft, field)

namespace ov::intel_cpu {

template <cpu::x64::cpu_isa_t isa>
jit_uni_dft_kernel_f32<isa>::jit_uni_dft_kernel_f32() : jit_uni_dft_kernel(),
                                                        jit_generator(jit_name()) {}

template <cpu::x64::cpu_isa_t isa>
void jit_uni_dft_kernel_f32<isa>::create_ker() {
    jit_generator::create_kernel();
    ker_ = (decltype(ker_))jit_ker();
}

template <cpu::x64::cpu_isa_t isa>
void jit_uni_dft_kernel_f32<isa>::generate() {
    this->preamble();

    mov(reg_src, ptr[reg_params + GET_OFF_DFT(src)]);
    mov(reg_dst, ptr[reg_params + GET_OFF_DFT(dst)]);
    mov(reg_twiddles, ptr[reg_params + GET_OFF_DFT(twiddles)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF_DFT(work_amount)]);
    mov(reg_index, ptr[reg_params + GET_OFF_DFT(index)]);

    Xbyak::Label main_loop_label;
    Xbyak::Label main_loop_end_label;
    Xbyak::Label tail_loop_label;
    Xbyak::Label tail_loop_end_label;

    uni_vpxor(vmm_sum, vmm_sum, vmm_sum);

    int step = vlen / 8;

    L(main_loop_label);
    {
        cmp(reg_work_amount, step);
        jl(main_loop_end_label, T_NEAR);

        uni_vmovups(vmm_data_cache, ptr[reg_src]);
        uni_vmovups(vmm_twiddles_cache, ptr[reg_twiddles]);

        uni_vshufps(vmm_data, vmm_data_cache, vmm_data_cache, 0b01000001);
        uni_vshufps(vmm_twiddles, vmm_twiddles_cache, vmm_twiddles_cache, 0b01000100);
        uni_vfmadd231ps(vmm_sum, vmm_data, vmm_twiddles);

        uni_vshufps(vmm_data, vmm_data_cache, vmm_data_cache, 0b11101011);
        uni_vshufps(vmm_twiddles, vmm_twiddles_cache, vmm_twiddles_cache, 0b11101110);
        uni_vfmadd231ps(vmm_sum, vmm_data, vmm_twiddles);

        add(reg_twiddles, 2 * step * sizeof(float));
        add(reg_src, 2 * step * sizeof(float));

        sub(reg_work_amount, step);
        jmp(main_loop_label, T_NEAR);
    }
    L(main_loop_end_label);

    if (mayiuse(cpu::x64::avx512_core)) {
        auto zmm_sum = Xbyak::Zmm(vmm_sum.getIdx());
        auto ymm_sum = Xbyak::Ymm(vmm_sum.getIdx());
        auto ymm_sum_2 = Xbyak::Ymm(vmm_sum_2.getIdx());

        vextractf64x4(ymm_sum_2, zmm_sum, 1);
        vaddps(ymm_sum, ymm_sum, ymm_sum_2);
    }
    if (mayiuse(cpu::x64::avx2)) {
        auto ymm_sum = Xbyak::Ymm(vmm_sum.getIdx());

        vextractf128(xmm_sum_2, ymm_sum, 1);
        vaddps(xmm_sum, xmm_sum, xmm_sum_2);
    }

    L(tail_loop_label);
    {
        cmp(reg_work_amount, 1);
        jl(tail_loop_end_label, T_NEAR);

        uni_vmovups(xmm_data, ptr[reg_src]);
        uni_vmovups(xmm_twiddles, ptr[reg_twiddles]);
        uni_vshufps(xmm_data, xmm_data, xmm_data, 0b01000001);
        uni_vshufps(xmm_twiddles, xmm_twiddles, xmm_twiddles, 0b01000100);
        uni_vfmadd231ps(xmm_sum, xmm_data, xmm_twiddles);

        add(reg_twiddles, 2 * sizeof(float));
        add(reg_src, 2 * sizeof(float));

        sub(reg_work_amount, 1);
        jmp(tail_loop_label, T_NEAR);
    }
    L(tail_loop_end_label);

    uni_vmovhlps(xmm_sum_2, xmm_sum_2, xmm_sum);
    uni_vhsubps(xmm_sum_2, xmm_sum_2, xmm_sum_2);
    uni_vhaddps(xmm_sum, xmm_sum, xmm_sum);

    uni_vmovss(ptr[reg_dst], xmm_sum_2);
    uni_vmovss(ptr[reg_dst + sizeof(float)], xmm_sum);

    this->postamble();
}

template struct jit_uni_dft_kernel_f32<cpu::x64::sse41>;
template struct jit_uni_dft_kernel_f32<cpu::x64::avx2>;
template struct jit_uni_dft_kernel_f32<cpu::x64::avx512_core>;

template <cpu::x64::cpu_isa_t isa>
jit_uni_fft_kernel_f32<isa>::jit_uni_fft_kernel_f32() : jit_uni_fft_kernel(),
                                                        jit_generator(jit_name()) {}

template <cpu::x64::cpu_isa_t isa>
void jit_uni_fft_kernel_f32<isa>::create_ker() {
    jit_generator::create_kernel();
    ker_ = (decltype(ker_))jit_ker();
}

template <cpu::x64::cpu_isa_t isa>
void jit_uni_fft_kernel_f32<isa>::generate() {
    this->preamble();

    mov(reg_src, ptr[reg_params + GET_OFF_FFT(src)]);
    mov(reg_dst, ptr[reg_params + GET_OFF_FFT(dst)]);
    mov(reg_twiddles_addr, ptr[reg_params + GET_OFF_FFT(twiddles)]);

    mov(reg_num_blocks, ptr[reg_params + GET_OFF_FFT(num_blocks)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF_FFT(work_amount)]);

    mov(reg_even_in_diff, sizeof(float));
    mul(ptr[reg_params + GET_OFF_FFT(n_complex)]);
    mov(reg_even_out_diff, reg_even_in_diff);

    mov(reg_even_in_diff, sizeof(float));
    mul(reg_work_amount);

    Xbyak::Label block_loop_label;
    Xbyak::Label block_loop_end_label;

    L(block_loop_label);
    {
        cmp(reg_num_blocks, 1);
        jl(block_loop_end_label, T_NEAR);

        mov(aux_reg_work_amount, reg_work_amount);
        uni_vbroadcastss(vmm_twiddle_real, ptr[reg_twiddles_addr]);
        uni_vbroadcastss(vmm_twiddle_imag, ptr[reg_twiddles_addr + sizeof(float)]);

        if (mayiuse(cpu::x64::avx2)) {
            loop_process<Vmm>(vlen / 4);
        }
        loop_process<Xbyak::Xmm>(4);
        loop_process<Xbyak::Xmm>(2);

        add(reg_twiddles_addr, 2 * sizeof(float));
        add(reg_src, reg_even_in_diff);
        sub(reg_num_blocks, 1);

        jmp(block_loop_label, T_NEAR);
    }
    L(block_loop_end_label);

    this->postamble();
}

template <cpu::x64::cpu_isa_t isa>
template <typename T>
void jit_uni_fft_kernel_f32<isa>::loop_process(int step) {
    T reg_data_odd_1 = T(vmm_data_odd_1.getIdx());
    T reg_data_odd_2 = T(vmm_data_odd_2.getIdx());
    T reg_twiddle_imag = T(vmm_twiddle_imag.getIdx());
    T reg_twiddle_real = T(vmm_twiddle_real.getIdx());
    T reg_data_even = T(vmm_data_even.getIdx());
    T reg_data_result = T(vmm_data_result.getIdx());

    Xbyak::Label loop_label;
    Xbyak::Label loop_end_label;

    L(loop_label);
    {
        cmp(aux_reg_work_amount, step);
        jl(loop_end_label, T_NEAR);

        move_data(reg_data_odd_1, ptr[reg_src + reg_even_in_diff], step);
        uni_vshufps(reg_data_odd_2, reg_data_odd_1, reg_data_odd_1, 0b10110001);
        uni_vmulps(reg_data_odd_2, reg_data_odd_2, reg_twiddle_imag);

        if (mayiuse(cpu::x64::avx512_core)) {
            vfmaddsub213ps(reg_data_odd_1, reg_twiddle_real, reg_data_odd_2);
        } else {
            uni_vmulps(reg_data_odd_1, reg_data_odd_1, reg_twiddle_real);
            uni_vaddsubps(reg_data_odd_1, reg_data_odd_1, reg_data_odd_2);
        }

        move_data(reg_data_even, ptr[reg_src], step);

        uni_vaddps(reg_data_result, reg_data_even, reg_data_odd_1);
        move_data(ptr[reg_dst], reg_data_result, step);

        uni_vsubps(reg_data_result, reg_data_even, reg_data_odd_1);
        move_data(ptr[reg_dst + reg_even_out_diff], reg_data_result, step);

        add(reg_src, step * sizeof(float));
        add(reg_dst, step * sizeof(float));

        sub(aux_reg_work_amount, step);
        jmp(loop_label, T_NEAR);
    }
    L(loop_end_label);
}

template <cpu::x64::cpu_isa_t isa>
void jit_uni_fft_kernel_f32<isa>::move_data(const Xbyak::Address& addr, const Xbyak::Xmm& x, int count) {
    if (count == 2) {
        uni_vmovq(addr, x);
    } else {
        uni_vmovups(addr, x);
    }
}

template <cpu::x64::cpu_isa_t isa>
void jit_uni_fft_kernel_f32<isa>::move_data(const Xbyak::Xmm& x, const Xbyak::Address& addr, int count) {
    if (count == 2) {
        uni_vmovq(x, addr);
    } else {
        uni_vmovups(x, addr);
    }
}

template struct jit_uni_fft_kernel_f32<cpu::x64::sse41>;
template struct jit_uni_fft_kernel_f32<cpu::x64::avx2>;
template struct jit_uni_fft_kernel_f32<cpu::x64::avx512_core>;

}  // namespace ov::intel_cpu
