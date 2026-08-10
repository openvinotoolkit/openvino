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
    vbroadcastss(v_input_scale, ptr[reg_x]);
    vmulps(v_input_scale, v_input_scale, v_delta);
    uni_vpxor(x_acc, x_acc, x_acc);

    const size_t vec_count = m_jcp.state_size / vec_size;
    for (size_t i = 0; i < vec_count; ++i) {
        const size_t offset = i * vec_bytes;
        vmovups(v_state, ptr[reg_state_src + offset]);
        vmovups(v_B, ptr[reg_B + offset]);
        vmulps(v_state, v_state, v_decay);
        vmulps(v_B, v_B, v_input_scale);
        vaddps(v_state, v_state, v_B);
        vmovups(ptr[reg_state_dst + offset], v_state);
        vmovups(v_C, ptr[reg_C + offset]);
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
        const size_t offset = (vectorized + i * 4) * sizeof(float);
        vmovups(Xbyak::Xmm(v_state.getIdx()), ptr[reg_state_src + offset]);
        vmovups(Xbyak::Xmm(v_B.getIdx()), ptr[reg_B + offset]);
        vmulps(Xbyak::Xmm(v_state.getIdx()), Xbyak::Xmm(v_state.getIdx()), Xbyak::Xmm(v_decay.getIdx()));
        vmulps(Xbyak::Xmm(v_B.getIdx()), Xbyak::Xmm(v_B.getIdx()), Xbyak::Xmm(v_input_scale.getIdx()));
        vaddps(Xbyak::Xmm(v_state.getIdx()), Xbyak::Xmm(v_state.getIdx()), Xbyak::Xmm(v_B.getIdx()));
        vmovups(ptr[reg_state_dst + offset], Xbyak::Xmm(v_state.getIdx()));
        vmovups(Xbyak::Xmm(v_C.getIdx()), ptr[reg_C + offset]);
        vmulps(Xbyak::Xmm(v_C.getIdx()), Xbyak::Xmm(v_C.getIdx()), Xbyak::Xmm(v_state.getIdx()));
        vaddps(x_acc, x_acc, Xbyak::Xmm(v_C.getIdx()));
    }

    const size_t scalar_begin = vectorized + four_count * 4;
    const size_t scalar_count = m_jcp.state_size - scalar_begin;
    for (size_t i = 0; i < scalar_count; ++i) {
        const size_t offset = (scalar_begin + i) * sizeof(float);
        vmovss(x_state, ptr[reg_state_src + offset]);
        vmovss(x_B, ptr[reg_B + offset]);
        vmulss(x_state, x_state, Xbyak::Xmm(v_decay.getIdx()));
        vmulss(x_B, x_B, Xbyak::Xmm(v_input_scale.getIdx()));
        vaddss(x_state, x_state, x_B);
        vmovss(ptr[reg_state_dst + offset], x_state);
        vmovss(x_C, ptr[reg_C + offset]);
        vmulss(x_C, x_C, x_state);
        uni_vpxor(x_tmp, x_tmp, x_tmp);
        vinsertps(x_tmp, x_tmp, x_C, static_cast<uint8_t>(i << 4));
        vaddps(x_acc, x_acc, x_tmp);
    }

    vpermilps(x_tmp, x_acc, 0xB1);
    vaddps(x_sum, x_acc, x_tmp);
    vpermilps(x_tmp, x_sum, 0x4E);
    vaddss(x_sum, x_sum, x_tmp);
    vmovss(ptr[reg_output], x_sum);

    add(reg_state_src, m_jcp.state_size * sizeof(float));
    add(reg_state_dst, m_jcp.state_size * sizeof(float));
    add(reg_x, sizeof(float));
    add(reg_output, sizeof(float));
    dec(reg_p_count);
    jnz(p_loop, T_NEAR);

    L(p_end);
    postamble();
}

std::shared_ptr<JitKernelBase> create_selective_ssm_jit_kernel(size_t state_size, bool prefer_avx512) {
    if (state_size == 0) {
        return nullptr;
    }

    jit_selective_ssm_compile_params jcp{state_size};
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
