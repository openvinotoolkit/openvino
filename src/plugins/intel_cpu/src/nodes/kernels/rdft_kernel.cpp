// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rdft_kernel.hpp"
#include <ie_common.h>

namespace ov {
namespace intel_cpu {

#define GET_OFF(field) offsetof(jit_dft_args, field)

template <cpu_isa_t isa>
void jit_dft_kernel_f32<isa>::generate() {
    using namespace Xbyak::util;
    using Xbyak::Label;
    using Xbyak::Xmm;
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm,
                                      isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

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
    }
    int vlen = cpu_isa_traits<isa>::vlen;
    const int simd_size = vlen / output_type_size;

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
    Xmm xmm_signal_size = Xmm(reg_idx);
    Vmm vmm_signal_size = Vmm(reg_idx);
    if (is_inverse_) {
        reg_idx++;
        uni_vbroadcastss(Vmm(reg_idx), ptr[param1 + GET_OFF(signal_size)]);
        uni_vcvtdq2ps(vmm_signal_size, Vmm(reg_idx));
    }

    Vmm vmm_neg_mask = Vmm(reg_idx);
    Xmm xmm_neg_mask = Xmm(reg_idx);
    if (kernel_type_ == complex_to_complex) {
        reg_idx++;
        if (!is_inverse_) {
            mov(rax, 1ULL << 31);
        } else {
            mov(rax, 1ULL << 63);
        }
        uni_vmovq(xmm_neg_mask, rax);
        uni_vbroadcastsd(vmm_neg_mask, xmm_neg_mask);
    }

    mov(rax, signal_size);
    and_(rax, 1);
    setz(is_signal_size_even);

    Label loop_over_output;
    Label loop_over_output_continue;
    Label loop_simd;
    Label loop_nonsimd;

    auto simd_loop = [this, vlen, simd_size,
                      input_type_size, reg_idx,
                      &vmm_signal_size,
                      &xmm_neg_mask,
                      &vmm_neg_mask] {
        size_t idx = reg_idx;
        Vmm result = Vmm(idx++);
        Vmm inp_real = Vmm(idx++);
        Vmm inp_imag = Vmm(idx++);
        const Vmm& input = inp_real;
        const Vmm& input_perm = inp_imag;
        Vmm twiddles = Vmm(idx++);
        const Vmm& cos = twiddles;
        Vmm sin = Vmm(idx++);
        Xmm tmp = Xmm(idx++);

        uni_vpxor(result, result, result);

        if (kernel_type_ == complex_to_complex && is_inverse_) {
            mov(rdx, 1ULL << 63);
            uni_vmovq(xmm_neg_mask, rdx);
            uni_vbroadcastsd(vmm_neg_mask, xmm_neg_mask);
        }

        Label loop;
        L(loop);
        {
            if (kernel_type_ == real_to_complex) {
                uni_vbroadcastss(inp_real, ptr[input_ptr]);
                uni_vmovups(twiddles, ptr[twiddles_ptr]);
                uni_vfmadd231ps(result, inp_real, twiddles);

                add(twiddles_ptr, vlen);
            } else if (kernel_type_ == complex_to_real) {
                uni_vbroadcastss(inp_real, ptr[input_ptr]);
                uni_vbroadcastss(inp_imag, ptr[input_ptr + type_size]);
                uni_vmovups(cos, ptr[twiddles_ptr]);
                uni_vmovups(sin, ptr[twiddles_ptr + vlen]);
                uni_vfmadd231ps(result, inp_real, cos);
                uni_vfmadd231ps(result, inp_imag, sin);

                add(twiddles_ptr, 2 * vlen);
            } else if (kernel_type_ == complex_to_complex) {
                // output_real += input_real * cos(..) - input_imag * sin(..)
                // output_imag += input_imag * cos(..) + input_real * sin(..)
                uni_vbroadcastsd(input, ptr[input_ptr]);
                uni_vpermilps(input_perm, input, 0b10110001); // swap real with imag
                uni_vpxor(input_perm, input_perm, vmm_neg_mask); // negate imag part (or real part if is_inverse == true)
                load_and_broadcast_every_other_elem(cos, twiddles_ptr, tmp);
                load_and_broadcast_every_other_elem(sin, twiddles_ptr + vlen / 2, tmp);
                uni_vfmadd231ps(result, input, cos);
                uni_vfmadd231ps(result, input_perm, sin);

                add(twiddles_ptr, vlen);
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

            if (kernel_type_ == complex_to_complex) {
                mov(rdx, 1ULL << 31);
                uni_vmovq(xmm_neg_mask, rdx);
                uni_vbroadcastsd(vmm_neg_mask, xmm_neg_mask);
            }

            test(is_signal_size_even, 1);
            jz(loop_backwards);

            sub(input_ptr, input_type_size);

            L(loop_backwards);
            {
                cmp(input_size, 0);
                je(loop_backwards_exit, T_NEAR);

                sub(input_ptr, input_type_size);
                if (kernel_type_ == complex_to_real) {
                    uni_vbroadcastss(inp_real, ptr[input_ptr]);
                    uni_vbroadcastss(inp_imag, ptr[input_ptr + type_size]);
                    uni_vmovups(cos, ptr[twiddles_ptr]);
                    uni_vmovups(sin, ptr[twiddles_ptr + vlen]);

                    uni_vfmadd231ps(result, inp_real, cos);
                    uni_vfnmadd231ps(result, inp_imag, sin);
                    add(twiddles_ptr, 2 * vlen);
                } else if (kernel_type_ == complex_to_complex) {
                    // output_real += input_real * cos(..) - input_imag * sin(..)
                    // output_imag += input_imag * cos(..) + input_real * sin(..)
                    uni_vbroadcastsd(input, ptr[input_ptr]);
                    uni_vpermilps(input_perm, input, 0b10110001); // swap real with imag
                    uni_vpxor(input_perm, input_perm, vmm_neg_mask); // negate imag part
                    load_and_broadcast_every_other_elem(cos, twiddles_ptr, tmp);
                    load_and_broadcast_every_other_elem(sin, twiddles_ptr + vlen / 2, tmp);
                    uni_vfmadd231ps(result, input, cos);
                    uni_vfmadd231ps(result, input_perm, sin);
                    add(twiddles_ptr, vlen);
                }

                dec(input_size);
                jmp(loop_backwards, T_NEAR);
            }
            L(loop_backwards_exit);
        }

        if (is_inverse_) {
            uni_vdivps(result, result, vmm_signal_size);
        }
        // store the results
        uni_vmovups(ptr[output_ptr], result);

        add(output_ptr, vlen);
        sub(output_end, simd_size);
    };

    auto nonsimd_loop = [this,
                         input_type_size,
                         output_type_size,
                         &xmm_signal_size,
                         reg_idx] {
        size_t idx = reg_idx;
        Xmm xmm_inp_real = Xbyak::Xmm(idx++);
        Xmm xmm_inp_imag = Xbyak::Xmm(idx++);
        Xmm xmm_real = Xbyak::Xmm(idx++);
        Xmm xmm_imag = Xbyak::Xmm(idx++);
        Xmm xmm_cos = Xbyak::Xmm(idx++);
        Xmm xmm_sin = Xbyak::Xmm(idx++);

        if (kernel_type_ != complex_to_real) {
            xorps(xmm_real, xmm_real);
            xorps(xmm_imag, xmm_imag);
        } else {
            xorps(xmm_real, xmm_real);
        }

        Label loop;
        L(loop);
        {
            movss(xmm_cos, ptr[twiddles_ptr]);
            movss(xmm_sin, ptr[twiddles_ptr + type_size]);
            if (kernel_type_ == real_to_complex) {
                movss(xmm_inp_real, ptr[input_ptr]);

                // output_real += input_real * cos(..)
                mulss(xmm_cos, xmm_inp_real);
                addss(xmm_real, xmm_cos);

                // output_imag += input_real * sin(..)
                mulss(xmm_sin, xmm_inp_real);
                addss(xmm_imag, xmm_sin);
            } else if (kernel_type_ == complex_to_real) {
                movss(xmm_inp_real, ptr[input_ptr]);
                movss(xmm_inp_imag, ptr[input_ptr + type_size]);

                // output += real * cos(..) + imag * sin(..)
                mulss(xmm_cos, xmm_inp_real);
                mulss(xmm_sin, xmm_inp_imag);
                addss(xmm_cos, xmm_sin);
                addss(xmm_real, xmm_cos);
            } else if (kernel_type_ == complex_to_complex) {
                // output_real += input_real * cos(..) - input_imag * sin(..)
                movss(xmm_inp_real, ptr[input_ptr]);
                movss(xmm_inp_imag, ptr[input_ptr + type_size]);
                mulss(xmm_inp_real, xmm_cos);
                mulss(xmm_inp_imag, xmm_sin);
                if (!is_inverse_) {
                    subss(xmm_inp_real, xmm_inp_imag);
                } else {
                    addss(xmm_inp_real, xmm_inp_imag);
                }
                addss(xmm_real, xmm_inp_real);

                // output_imag += input_imag * cos(..) + input_real * sin(..)
                movss(xmm_inp_real, ptr[input_ptr]);
                movss(xmm_inp_imag, ptr[input_ptr + type_size]);
                mulss(xmm_inp_imag, xmm_cos);
                mulss(xmm_inp_real, xmm_sin);
                if (!is_inverse_) {
                    addss(xmm_inp_imag, xmm_inp_real);
                } else {
                    subss(xmm_inp_imag, xmm_inp_real);
                }
                addss(xmm_imag, xmm_inp_imag);
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

                movss(xmm_cos, ptr[twiddles_ptr]);
                movss(xmm_sin, ptr[twiddles_ptr + type_size]);
                movss(xmm_inp_real, ptr[input_ptr]);
                movss(xmm_inp_imag, ptr[input_ptr + type_size]);

                if (kernel_type_ == complex_to_real) {
                    // output += real * cos(..) - imag * sin(..)
                    mulss(xmm_cos, xmm_inp_real);
                    mulss(xmm_sin, xmm_inp_imag);
                    subss(xmm_cos, xmm_sin);
                    addss(xmm_real, xmm_cos);
                } else if (kernel_type_ == complex_to_complex) {
                    // output_real += input_real * cos(..) - input_imag * sin(..)
                    movss(xmm_inp_real, ptr[input_ptr]);
                    movss(xmm_inp_imag, ptr[input_ptr + type_size]);
                    mulss(xmm_inp_real, xmm_cos);
                    mulss(xmm_inp_imag, xmm_sin);
                    subss(xmm_inp_real, xmm_inp_imag);
                    addss(xmm_real, xmm_inp_real);

                    // output_imag += input_imag * cos(..) + input_real * sin(..)
                    movss(xmm_inp_real, ptr[input_ptr]);
                    movss(xmm_inp_imag, ptr[input_ptr + type_size]);
                    mulss(xmm_inp_imag, xmm_cos);
                    mulss(xmm_inp_real, xmm_sin);
                    addss(xmm_inp_imag, xmm_inp_real);
                    addss(xmm_imag, xmm_inp_imag);
                }

                add(twiddles_ptr, complex_type_size<float>());
                dec(input_size);
                jmp(loop_backwards);
            }
            L(loop_backwards_exit);
        }

        if (kernel_type_ == complex_to_real) {
            if (is_inverse_) {
                divss(xmm_real, xmm_signal_size);
            }
            // store the result
            movss(ptr[output_ptr], xmm_real);
        } else {
            if (is_inverse_) {
                divss(xmm_real, xmm_signal_size);
                divss(xmm_imag, xmm_signal_size);
            }
            // store the results
            movss(ptr[output_ptr], xmm_real);
            movss(ptr[output_ptr + type_size], xmm_imag);
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

template <cpu_isa_t isa>
void jit_dft_kernel_f32<isa>::uni_vbroadcastsd(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
    movsd(x, op);
    shufpd(x, x, 0x0);
}

template <cpu_isa_t isa>
void jit_dft_kernel_f32<isa>::uni_vbroadcastsd(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
    vbroadcastsd(x, op);
}

template <cpu_isa_t isa>
void jit_dft_kernel_f32<isa>::uni_vpermilps(const Xbyak::Xmm& x, const Xbyak::Operand& op, int8_t control) {
    movups(x, op);
    shufps(x, x, control);
}

template <cpu_isa_t isa>
void jit_dft_kernel_f32<isa>::uni_vpermilps(const Xbyak::Ymm& x, const Xbyak::Operand& op, int8_t control) {
    vpermilps(x, op, control);
}

template <cpu_isa_t isa>
void jit_dft_kernel_f32<isa>::load_and_broadcast_every_other_elem(const Xbyak::Zmm& x, const Xbyak::RegExp& reg_exp, const Xbyak::Xmm& tmp) {
    for (int i = 0; i < 4; i++) {
        movq(tmp, ptr[reg_exp + type_size * i * 2]);
        shufps(tmp, tmp, 0b01010000);
        vinsertf32x4(x, x, tmp, i);
    }
}

template <cpu_isa_t isa>
void jit_dft_kernel_f32<isa>::load_and_broadcast_every_other_elem(const Xbyak::Ymm& x, const Xbyak::RegExp& reg_exp, const Xbyak::Xmm& tmp) {
    for (int i = 0; i < 2; i++) {
        movq(tmp, ptr[reg_exp + type_size * i * 2]);
        shufps(tmp, tmp, 0b01010000);
        vinsertf128(x, x, tmp, i);
    }
}

template <cpu_isa_t isa>
void jit_dft_kernel_f32<isa>::load_and_broadcast_every_other_elem(const Xbyak::Xmm& x, const Xbyak::RegExp& reg_exp, const Xbyak::Xmm& tmp) {
    movq(x, ptr[reg_exp]);
    shufps(x, x, 0b01010000);
}

template struct jit_dft_kernel_f32<cpu::x64::sse41>;
template struct jit_dft_kernel_f32<cpu::x64::avx2>;
template struct jit_dft_kernel_f32<cpu::x64::avx512_core>;

}   // namespace intel_cpu
}   // namespace ov
