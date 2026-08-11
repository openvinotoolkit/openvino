// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_jit_kernel.hpp"

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>

using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::kernel {

#define GET_OFF(field) offsetof(jit_selective_ssm_call_args, field)

template <cpu_isa_t isa>
size_t jit_selective_ssm_kernel<isa>::data_size() const {
    return m_jcp.data_type == jit_selective_ssm_data_type::f32 ? sizeof(float) : sizeof(uint16_t);
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::load_data_vector(const Vmm& dst, const Xbyak::Reg64& base, size_t element_offset) {
    const auto offset = element_offset * data_size();
    if (m_jcp.data_type == jit_selective_ssm_data_type::f32) {
        vmovups(dst, ptr[base + offset]);
    } else if (m_jcp.data_type == jit_selective_ssm_data_type::f16) {
        if constexpr (isa == avx2) {
            vcvtph2ps(dst, xword[base + offset]);
        } else {
            vcvtph2ps(dst, yword[base + offset]);
        }
    } else {
        if constexpr (isa == avx2) {
            vpmovzxwd(dst, xword[base + offset]);
        } else {
            vpmovzxwd(dst, yword[base + offset]);
        }
        vpslld(dst, dst, 16);
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::load_data_xmm(const Xbyak::Xmm& dst,
                                                  const Xbyak::Reg64& base,
                                                  size_t element_offset) {
    const auto offset = element_offset * data_size();
    if (m_jcp.data_type == jit_selective_ssm_data_type::f32) {
        vmovups(dst, ptr[base + offset]);
    } else if (m_jcp.data_type == jit_selective_ssm_data_type::f16) {
        vcvtph2ps(dst, qword[base + offset]);
    } else {
        vpmovzxwd(dst, qword[base + offset]);
        vpslld(dst, dst, 16);
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::load_data_scalar(const Xbyak::Xmm& dst,
                                                     const Xbyak::Reg64& base,
                                                     size_t element_offset) {
    const auto offset = element_offset * data_size();
    if (m_jcp.data_type == jit_selective_ssm_data_type::f32) {
        vmovss(dst, ptr[base + offset]);
        return;
    }

    movzx(reg_tmp.cvt32(), word[base + offset]);
    if (m_jcp.data_type == jit_selective_ssm_data_type::bf16) {
        shl(reg_tmp.cvt32(), 16);
    }
    vmovd(dst, reg_tmp.cvt32());
    if (m_jcp.data_type == jit_selective_ssm_data_type::f16) {
        vcvtph2ps(dst, dst);
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::broadcast_data(const Vmm& dst, const Xbyak::Reg64& base) {
    if (m_jcp.data_type == jit_selective_ssm_data_type::f32) {
        vbroadcastss(dst, ptr[base]);
    } else {
        load_data_scalar(x_tmp, base, 0);
        vbroadcastss(dst, x_tmp);
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::store_data_scalar(const Xbyak::Reg64& base, const Xbyak::Xmm& value) {
    if (m_jcp.data_type == jit_selective_ssm_data_type::f32) {
        vmovss(ptr[base], value);
    } else if (m_jcp.data_type == jit_selective_ssm_data_type::f16) {
        vcvtps2ph(x_tmp, value, 0x4);
        vmovd(reg_tmp.cvt32(), x_tmp);
        mov(word[base], reg_tmp.cvt16());
    } else {
        vmovd(reg_tmp.cvt32(), value);
        mov(reg_round.cvt32(), reg_tmp.cvt32());
        and_(reg_round.cvt32(), 0x00010000);
        shr(reg_round.cvt32(), 1);
        add(reg_tmp.cvt32(), reg_round.cvt32());
        shr(reg_tmp.cvt32(), 16);
        mov(word[base], reg_tmp.cvt16());
    }
}

template <cpu_isa_t isa>
void jit_selective_ssm_kernel<isa>::generate() {
    preamble();

    mov(reg_args, abi_param1);
    mov(reg_state_src, ptr[reg_args + GET_OFF(state_src)]);
    mov(reg_state_dst, ptr[reg_args + GET_OFF(state_dst)]);
    mov(reg_B, ptr[reg_args + GET_OFF(B)]);
    mov(reg_C, ptr[reg_args + GET_OFF(C)]);
    mov(reg_x, ptr[reg_args + GET_OFF(x)]);
    mov(reg_p_count, ptr[reg_args + GET_OFF(p_count)]);
    mov(reg_output, ptr[reg_args + GET_OFF(output)]);
    vbroadcastss(v_decay, ptr[reg_args + GET_OFF(decay)]);
    vbroadcastss(v_delta, ptr[reg_args + GET_OFF(delta)]);

    Xbyak::Label p_loop;
    Xbyak::Label p_end;
    test(reg_p_count, reg_p_count);
    jz(p_end, T_NEAR);

    L(p_loop);
    broadcast_data(v_input_scale, reg_x);
    vmulps(v_input_scale, v_input_scale, v_delta);
    uni_vpxor(x_acc, x_acc, x_acc);

    const size_t vec_count = m_jcp.state_size / vec_size;
    for (size_t i = 0; i < vec_count; ++i) {
        const size_t state_offset = i * vec_bytes;
        const size_t element_offset = i * vec_size;
        vmovups(v_state, ptr[reg_state_src + state_offset]);
        load_data_vector(v_B, reg_B, element_offset);
        vmulps(v_state, v_state, v_decay);
        if (m_jcp.data_type == jit_selective_ssm_data_type::f32) {
            vmulps(v_B, v_B, v_input_scale);
            vaddps(v_state, v_state, v_B);
        } else {
            vfmadd231ps(v_state, v_B, v_input_scale);
        }
        vmovups(ptr[reg_state_dst + state_offset], v_state);
        load_data_vector(v_C, reg_C, element_offset);
        vmulps(v_C, v_C, v_state);
        vaddps(x_acc, x_acc, Xbyak::Xmm(v_C.getIdx()));
        if constexpr (isa == avx2) {
            vextractf128(x_tmp, Xbyak::Ymm(v_C.getIdx()), 1);
            vaddps(x_acc, x_acc, x_tmp);
        } else {
            for (int quarter = 1; quarter < 4; ++quarter) {
                vextractf32x4(x_tmp, Xbyak::Zmm(v_C.getIdx()), quarter);
                vaddps(x_acc, x_acc, x_tmp);
            }
        }
    }

    const size_t vectorized = vec_count * vec_size;
    const size_t four_count = (m_jcp.state_size - vectorized) / 4;
    for (size_t i = 0; i < four_count; ++i) {
        const size_t element_offset = vectorized + i * 4;
        const size_t state_offset = element_offset * sizeof(float);
        vmovups(x_state, ptr[reg_state_src + state_offset]);
        load_data_xmm(Xbyak::Xmm(v_B.getIdx()), reg_B, element_offset);
        vmulps(x_state, x_state, Xbyak::Xmm(v_decay.getIdx()));
        if (m_jcp.data_type == jit_selective_ssm_data_type::f32) {
            vmulps(Xbyak::Xmm(v_B.getIdx()), Xbyak::Xmm(v_B.getIdx()), Xbyak::Xmm(v_input_scale.getIdx()));
            vaddps(x_state, x_state, Xbyak::Xmm(v_B.getIdx()));
        } else {
            vfmadd231ps(x_state, Xbyak::Xmm(v_B.getIdx()), Xbyak::Xmm(v_input_scale.getIdx()));
        }
        vmovups(ptr[reg_state_dst + state_offset], x_state);
        load_data_xmm(Xbyak::Xmm(v_C.getIdx()), reg_C, element_offset);
        vfmadd231ps(x_acc, Xbyak::Xmm(v_C.getIdx()), x_state);
    }

    const size_t scalar_begin = vectorized + four_count * 4;
    const size_t scalar_count = m_jcp.state_size - scalar_begin;
    for (size_t i = 0; i < scalar_count; ++i) {
        const size_t element_offset = scalar_begin + i;
        const size_t state_offset = element_offset * sizeof(float);
        vmovss(x_state, ptr[reg_state_src + state_offset]);
        load_data_scalar(x_B, reg_B, element_offset);
        vmulss(x_state, x_state, Xbyak::Xmm(v_decay.getIdx()));
        if (m_jcp.data_type == jit_selective_ssm_data_type::f32) {
            vmulss(x_B, x_B, Xbyak::Xmm(v_input_scale.getIdx()));
            vaddss(x_state, x_state, x_B);
        } else {
            vfmadd231ss(x_state, x_B, Xbyak::Xmm(v_input_scale.getIdx()));
        }
        vmovss(ptr[reg_state_dst + state_offset], x_state);
        load_data_scalar(x_C, reg_C, element_offset);
        vfmadd231ss(x_acc, x_C, x_state);
    }

    vpermilps(x_tmp, x_acc, 0xB1);
    vaddps(x_sum, x_acc, x_tmp);
    vpermilps(x_tmp, x_sum, 0x4E);
    vaddss(x_sum, x_sum, x_tmp);
    store_data_scalar(reg_output, x_sum);

    add(reg_state_src, m_jcp.state_size * sizeof(float));
    add(reg_state_dst, m_jcp.state_size * sizeof(float));
    add(reg_x, data_size());
    add(reg_output, data_size());
    dec(reg_p_count);
    jnz(p_loop, T_NEAR);

    L(p_end);
    postamble();
}

std::shared_ptr<JitKernelBase> create_selective_ssm_jit_kernel(size_t state_size,
                                                               bool prefer_avx512,
                                                               jit_selective_ssm_data_type data_type) {
    if (state_size == 0) {
        return nullptr;
    }

    jit_selective_ssm_compile_params jcp{state_size, data_type};
    std::shared_ptr<JitKernelBase> kernel;
    if (prefer_avx512 && mayiuse(avx512_core)) {
        kernel = std::make_shared<jit_selective_ssm_kernel<avx512_core>>(jcp);
    } else if (mayiuse(avx2)) {
        kernel = std::make_shared<jit_selective_ssm_kernel<avx2>>(jcp);
    }
    if (kernel) {
        kernel->create_kernel();
    }
    return kernel;
}

template struct jit_selective_ssm_kernel<avx2>;
template struct jit_selective_ssm_kernel<avx512_core>;

#undef GET_OFF

}  // namespace ov::intel_cpu::kernel
