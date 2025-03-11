// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rdft_kernel.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

#ifndef OPENVINO_ARCH_ARM64
#    define GET_OFF(field) offsetof(jit_dft_args, field)

template <cpu_isa_t isa>
void jit_dft_kernel_f32<isa>::generate() {
    using namespace Xbyak::util;
    using Xbyak::Label;
    using Xbyak::Xmm;

    this->preamble();

    int input_type_size = 0;
    int output_type_size = 0;

    switch (kernel_type_) {
    case real_to_complex:
        input_type_size = type_size;
        output_type_size = complex_type_size<float>();
        break;
    case complex_to_complex:
        input_type_size = complex_type_size<float>();
        output_type_size = complex_type_size<float>();
        break;
    case complex_to_real:
        input_type_size = complex_type_size<float>();
        output_type_size = type_size;
        break;
    default:
        output_type_size = 1;
        break;
    }
    int simd_size = vlen / output_type_size;
    if (kernel_type_ == complex_to_complex) {
        simd_size = vlen / type_size;
    }

    mov(input_ptr, ptr[param1 + GET_OFF(input)]);
    mov(input_size, ptr[param1 + GET_OFF(input_size)]);
    mov(twiddles_ptr, ptr[param1 + GET_OFF(twiddles)]);
    mov(output_start, ptr[param1 + GET_OFF(output_start)]);
    mov(output_end, ptr[param1 + GET_OFF(output_end)]);

    // offset twiddles_ptr by input_size * complex_type_size<float>() * output_start bytes
    mov(signal_size, ptr[param1 + GET_OFF(signal_size)]);
    mov(rax, signal_size);
    lea(rax, ptr[rax * complex_type_size<float>()]);
    xor_(rdx, rdx);
    mul(output_start);
    add(twiddles_ptr, rax);

    // offset output_ptr by output_start * output_type_size bytes
    mov(output_ptr, ptr[param1 + GET_OFF(output)]);
    lea(output_ptr, ptr[output_ptr + output_type_size * output_start]);

    size_t reg_idx = 0;
    auto xmm_signal_size = Xmm(reg_idx);
    auto vmm_signal_size = Vmm(reg_idx);
    if (is_inverse_) {
        reg_idx++;
        uni_vbroadcastss(vmm_signal_size, ptr[param1 + GET_OFF(signal_size)]);
        uni_vcvtdq2ps(vmm_signal_size, vmm_signal_size);
    }

    auto neg_mask = Xmm(reg_idx);
    if (kernel_type_ == complex_to_complex) {
        reg_idx++;
        uni_vpxor(neg_mask, neg_mask, neg_mask);
        mov(rax, 1ULL << 63);
        uni_vmovq(neg_mask, rax);
    }

    size_t vmm_reg_idx = reg_idx;
    auto inp_real = Vmm(vmm_reg_idx++);
    auto inp_imag = Vmm(vmm_reg_idx++);
    auto cos = Vmm(vmm_reg_idx++);
    auto sin = Vmm(vmm_reg_idx++);
    const Vmm& twiddles = cos;
    auto tmp = Vmm(vmm_reg_idx++);
    auto output_real = Vmm(vmm_reg_idx++);
    auto output_imag = Vmm(vmm_reg_idx++);
    const Vmm& output = output_real;
    perm_low = Vmm(vmm_reg_idx++);
    perm_high = Vmm(vmm_reg_idx++);

    mov(rax, reinterpret_cast<uint64_t>(perm_low_values.data()));
    uni_vmovups(perm_low, ptr[rax]);
    mov(rax, reinterpret_cast<uint64_t>(perm_high_values.data()));
    uni_vmovups(perm_high, ptr[rax]);

    auto xmm_input = Xbyak::Xmm(reg_idx++);
    auto xmm_twiddles = Xbyak::Xmm(reg_idx++);
    auto xmm_output = Xbyak::Xmm(reg_idx++);

    mov(rax, signal_size);
    and_(rax, 1);
    setz(is_signal_size_even);

    Label loop_over_output;
    Label loop_over_output_continue;
    Label loop_simd;
    Label loop_nonsimd;

    auto simd_loop = [&] {
        if (kernel_type_ == complex_to_complex) {
            uni_vpxor(output_real, output_real, output_real);
            uni_vpxor(output_imag, output_imag, output_imag);
        } else {
            uni_vpxor(output, output, output);
        }

        auto c2r_kernel = [&](bool backwards) {
            // if backwards == false:
            //     output_real += input_real * cos(..) - input_imag * sin(..)
            // else:
            //     output_real += input_real * cos(..) + input_imag * sin(..)
            uni_vbroadcastss(inp_real, ptr[input_ptr]);
            uni_vbroadcastss(inp_imag, ptr[input_ptr + type_size]);
            uni_vmovups(cos, ptr[twiddles_ptr]);
            uni_vmovups(sin, ptr[twiddles_ptr + vlen]);
            uni_vfmadd231ps(output, inp_real, cos);
            if (!backwards) {
                uni_vfnmadd231ps(output, inp_imag, sin);
            } else {
                uni_vfmadd231ps(output, inp_imag, sin);
            }
            add(twiddles_ptr, 2 * vlen);
        };

        auto c2c_kernel = [&](bool backwards) {
            // if backwards == false:
            //     output_real += input_real * cos(..) - input_imag * sin(..)
            //     output_imag += input_imag * cos(..) + input_real * sin(..)
            // else:
            //     output_real += input_real * cos(..) + input_imag * sin(..)
            //     output_imag += input_imag * cos(..) - input_real * sin(..)
            uni_vbroadcastss(inp_real, ptr[input_ptr]);
            uni_vbroadcastss(inp_imag, ptr[input_ptr + type_size]);
            uni_vmovups(cos, ptr[twiddles_ptr]);
            uni_vmovups(sin, ptr[twiddles_ptr + vlen]);
            uni_vfmadd231ps(output_real, inp_real, cos);
            uni_vfmadd231ps(output_imag, inp_imag, cos);
            if (!backwards) {
                uni_vfnmadd231ps(output_real, inp_imag, sin);
                uni_vfmadd231ps(output_imag, inp_real, sin);
            } else {
                uni_vfmadd231ps(output_real, inp_imag, sin);
                uni_vfnmadd231ps(output_imag, inp_real, sin);
            }

            add(twiddles_ptr, 2 * vlen);
        };

        Label loop;
        L(loop);
        {
            if (kernel_type_ == real_to_complex) {
                uni_vbroadcastss(inp_real, ptr[input_ptr]);
                uni_vmovups(twiddles, ptr[twiddles_ptr]);
                uni_vfmadd231ps(output, inp_real, twiddles);
                add(twiddles_ptr, vlen);
            } else if (kernel_type_ == complex_to_real) {
                c2r_kernel(false);
            } else if (kernel_type_ == complex_to_complex) {
                c2c_kernel(false);
            }

            add(input_ptr, input_type_size);

            dec(input_size);
            cmp(input_size, 0);
            jne(loop, T_NEAR);
        }

        if (is_inverse_) {
            Label loop_backwards;
            Label loop_backwards_exit;

            mov(input_size, signal_size);
            sub(input_size, ptr[param1 + GET_OFF(input_size)]);

            test(is_signal_size_even, 1);
            jz(loop_backwards);

            sub(input_ptr, input_type_size);

            L(loop_backwards);
            {
                cmp(input_size, 0);
                je(loop_backwards_exit, T_NEAR);

                sub(input_ptr, input_type_size);

                if (kernel_type_ == complex_to_real) {
                    c2r_kernel(true);
                } else if (kernel_type_ == complex_to_complex) {
                    c2c_kernel(true);
                }

                dec(input_size);
                jmp(loop_backwards, T_NEAR);
            }
            L(loop_backwards_exit);
        }

        if (is_inverse_) {
            uni_vdivps(output_real, output_real, vmm_signal_size);
            uni_vdivps(output_imag, output_imag, vmm_signal_size);
        }

        // store the results
        if (kernel_type_ == complex_to_complex) {
            interleave_and_store(output_real, output_imag, output_ptr, tmp);
            add(output_ptr, 2 * vlen);
        } else {
            uni_vmovups(ptr[output_ptr], output);
            add(output_ptr, vlen);
        }

        sub(output_end, simd_size);
    };

    auto nonsimd_loop = [&] {
        uni_vxorps(xmm_output, xmm_output, xmm_output);

        auto c2r_kernel = [&](bool backwards) {
            // if backwards == false:
            //     output_real += input_real * cos(..) - input_imag * sin(..)
            // else:
            //     output_real += input_real * cos(..) + input_imag * sin(..)
            uni_vmovq(xmm_input, ptr[input_ptr]);
            uni_vmovq(xmm_twiddles, ptr[twiddles_ptr]);
            uni_vmulps(xmm_input, xmm_input, xmm_twiddles);
            if (!backwards) {
                uni_vhsubps(xmm_input, xmm_input, xmm_input);
            } else {
                uni_vhaddps(xmm_input, xmm_input, xmm_input);
            }
            uni_vaddss(xmm_output, xmm_output, xmm_input);
        };

        auto c2c_kernel = [&](bool backwards) {
            // if backwards == false:
            //     output_real += input_real * cos(..) - input_imag * sin(..)
            //     output_imag += input_imag * cos(..) + input_real * sin(..)
            // else:
            //     output_real += input_real * cos(..) + input_imag * sin(..)
            //     output_imag += input_imag * cos(..) - input_real * sin(..)
            uni_vmovq(xmm_input, ptr[input_ptr]);
            uni_vshufps(xmm_input, xmm_input, xmm_input, 0b00010100);
            uni_vmovq(xmm_twiddles, ptr[twiddles_ptr]);
            uni_vshufps(xmm_twiddles, xmm_twiddles, xmm_twiddles, 0b01000100);
            uni_vxorps(xmm_twiddles, xmm_twiddles, neg_mask);
            uni_vmulps(xmm_input, xmm_input, xmm_twiddles);
            if (!backwards) {
                uni_vhaddps(xmm_input, xmm_input, xmm_input);
            } else {
                uni_vhsubps(xmm_input, xmm_input, xmm_input);
            }
            uni_vaddps(xmm_output, xmm_output, xmm_input);
        };

        Label loop;
        L(loop);
        {
            if (kernel_type_ == real_to_complex) {
                // output_real += input_real * cos(..)
                // output_imag += input_real * sin(..)
                uni_vmovq(xmm_twiddles, ptr[twiddles_ptr]);
                uni_vmovd(xmm_input, ptr[input_ptr]);
                uni_vshufps(xmm_input, xmm_input, xmm_input, 0);
                uni_vmulps(xmm_input, xmm_input, xmm_twiddles);
                uni_vaddps(xmm_output, xmm_output, xmm_input);
            } else if (kernel_type_ == complex_to_real) {
                c2r_kernel(false);
            } else if (kernel_type_ == complex_to_complex) {
                c2c_kernel(false);
            }

            // increment indexes for next iteration
            add(twiddles_ptr, complex_type_size<float>());
            add(input_ptr, input_type_size);
            dec(input_size);

            // continue if input_size > 0
            cmp(input_size, 0);
            jg(loop, T_NEAR);
        }
        if (is_inverse_) {
            Label loop_backwards;
            Label loop_backwards_exit;

            mov(input_size, signal_size);
            sub(input_size, ptr[param1 + GET_OFF(input_size)]);

            test(is_signal_size_even, 1);
            jz(loop_backwards);

            sub(input_ptr, input_type_size);

            L(loop_backwards);
            {
                cmp(input_size, 0);
                je(loop_backwards_exit);

                sub(input_ptr, input_type_size);

                if (kernel_type_ == complex_to_real) {
                    c2r_kernel(true);
                } else if (kernel_type_ == complex_to_complex) {
                    c2c_kernel(true);
                }

                add(twiddles_ptr, complex_type_size<float>());
                dec(input_size);
                jmp(loop_backwards);
            }
            L(loop_backwards_exit);
        }

        if (kernel_type_ == complex_to_real) {
            if (is_inverse_) {
                uni_vdivss(xmm_output, xmm_output, xmm_signal_size);
            }
            // store the result
            uni_vmovss(ptr[output_ptr], xmm_output);
        } else {
            if (is_inverse_) {
                uni_vdivps(xmm_output, xmm_output, xmm_signal_size);
            }
            // store the results
            uni_vmovq(ptr[output_ptr], xmm_output);
        }

        add(output_ptr, output_type_size);
        dec(output_end);
    };

    L(loop_over_output);
    {
        mov(input_ptr, ptr[param1 + GET_OFF(input)]);
        mov(input_size, ptr[param1 + GET_OFF(input_size)]);

        cmp(output_end, simd_size);
        jae(loop_simd, T_NEAR);

        jmp(loop_nonsimd, T_NEAR);

        L(loop_simd);
        simd_loop();
        jmp(loop_over_output_continue, T_NEAR);

        L(loop_nonsimd);
        nonsimd_loop();

        L(loop_over_output_continue);
        cmp(output_end, 0);
        ja(loop_over_output, T_NEAR);
    }

    this->postamble();
}

// Interleave real and imag registers and store in memory.
// For example (for AVX):
// real = [1, 2, 3, 4, 5, 6, 7, 8]
// imag = [11, 12, 13, 14, 15, 16, 17, 18]
// interleaved = [1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18]
template <>
void jit_dft_kernel_f32<avx512_core>::interleave_and_store(const Vmm& real,
                                                           const Vmm& imag,
                                                           const Xbyak::RegExp& reg_exp,
                                                           const Vmm& tmp) {
    const Vmm& low = tmp;
    const Vmm& high = real;
    uni_vmovups(low, real);
    vpermt2ps(low, perm_low, imag);
    vpermt2ps(high, perm_high, imag);
    uni_vmovups(ptr[reg_exp], low);
    uni_vmovups(ptr[reg_exp + vlen], high);
}

template <>
void jit_dft_kernel_f32<avx2>::interleave_and_store(const Vmm& real,
                                                    const Vmm& imag,
                                                    const Xbyak::RegExp& reg_exp,
                                                    const Vmm& tmp) {
    const Vmm& low = real;
    const Vmm& high = imag;
    vunpcklps(tmp, real, imag);
    vunpckhps(high, real, imag);
    vinsertf128(low, tmp, Xbyak::Xmm(high.getIdx()), 1);
    vperm2f128(high, tmp, high, 0b00110001);
    uni_vmovups(ptr[reg_exp], low);
    uni_vmovups(ptr[reg_exp + vlen], high);
}

template <>
void jit_dft_kernel_f32<sse41>::interleave_and_store(const Vmm& real,
                                                     const Vmm& imag,
                                                     const Xbyak::RegExp& reg_exp,
                                                     const Vmm& tmp) {
    const Vmm& low = tmp;
    const Vmm& high = real;
    uni_vmovups(low, real);
    unpcklps(low, imag);
    unpckhps(high, imag);
    uni_vmovups(ptr[reg_exp], low);
    uni_vmovups(ptr[reg_exp + vlen], high);
}

template struct jit_dft_kernel_f32<cpu::x64::sse41>;
template struct jit_dft_kernel_f32<cpu::x64::avx2>;
template struct jit_dft_kernel_f32<cpu::x64::avx512_core>;

#endif
}  // namespace ov::intel_cpu
