// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_kernel.hpp"

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace kernel {

#define GET_OFF(field) offsetof(jit_rms_call_args, field)

template <cpu_isa_t isa>
void jit_rms_kernel<isa>::reduce_zmm_to_ymm(
        const Xmm &acc, const Xmm &tmp) {
    const Zmm zmm_acc(acc.getIdx());
    const Ymm ymm_acc(acc.getIdx());
    const Ymm ymm_to_acc(tmp.getIdx());
    vextractf64x4(ymm_to_acc, zmm_acc, 1);
    vaddps(ymm_acc, ymm_acc, ymm_to_acc);
}

template <cpu_isa_t isa>
void jit_rms_kernel<isa>::reduce_ymm_to_xmm(
        const Xmm &acc, const Xmm &tmp) {
    const Ymm ymm_acc(acc.getIdx());
    const Xmm xmm_acc(acc.getIdx());
    const Xmm xmm_to_acc(tmp.getIdx());
    vextractf128(xmm_to_acc, ymm_acc, 1);
    vaddps(xmm_acc, xmm_acc, xmm_to_acc);
}

template <cpu_isa_t isa>
void jit_rms_kernel<isa>::reduce_xmm_to_scalar(const Xmm &acc,
        const Xmm &tmp, const std::size_t number_of_values_to_reduce) {
    assert(number_of_values_to_reduce <= number_of_f32_in_xmm_);

    const Xmm xmm_acc(acc.getIdx());
    const Xmm ymm_to_acc(tmp.getIdx());

    static constexpr int number_of_f32_to_move = number_of_f32_in_xmm_ - 1;
    static constexpr uint8_t insertps_configuration[number_of_f32_to_move]
            = {0b01001110, 0b10001110, 0b11001110};

    for (std::size_t i = 0; i < number_of_values_to_reduce - 1; i++) {
        vinsertps(ymm_to_acc, ymm_to_acc, xmm_acc, insertps_configuration[i]);
        vaddss(xmm_acc, xmm_acc, ymm_to_acc);
    }
}

template <cpu_isa_t isa>
void jit_rms_kernel<isa>::reduce_ymm_to_scalar(
        const Xbyak::Xmm &acc, const Xbyak::Xmm &tmp1, const Xbyak::Xmm &tmp2,
        const std::size_t number_of_values_to_reduce) {
    assert(number_of_values_to_reduce <= number_of_f32_in_ymm_);

    const Ymm ymm_acc(acc.getIdx());
    const Xmm xmm_acc(acc.getIdx());
    const Xmm xmm_tmp(tmp1.getIdx());
    const Xmm xmm_acc_upper_half(tmp2.getIdx());

    if (number_of_values_to_reduce == number_of_f32_in_ymm_) {
        reduce_ymm_to_xmm(ymm_acc, xmm_tmp);
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp);
    } else if (number_of_values_to_reduce > number_of_f32_in_xmm_) {
        vextractf128(xmm_acc_upper_half, ymm_acc, 1);
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp);
        reduce_xmm_to_scalar(xmm_acc_upper_half, xmm_tmp,
                number_of_values_to_reduce - number_of_f32_in_xmm_);
        vaddss(xmm_acc, xmm_acc, xmm_acc_upper_half);
    } else if (number_of_values_to_reduce <= number_of_f32_in_xmm_) {
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp, number_of_values_to_reduce);
    }
}

template <cpu_isa_t isa>
void jit_rms_kernel<isa>::reduce_vmm_to_scalar(
        const Xbyak::Xmm &acc, const Xbyak::Xmm &tmp1, const Xbyak::Xmm &tmp2,
        const Xbyak::Xmm &tmp3, const std::size_t number_of_values_to_reduce) {
    assert(number_of_values_to_reduce <= number_of_f32_in_zmm_);

    const Zmm zmm_acc(acc.getIdx());
    const Ymm ymm_acc(acc.getIdx());
    const Xmm xmm_acc(acc.getIdx());
    const Ymm ymm_acc_upper_half(tmp1.getIdx());
    const Xmm xmm_acc_upper_half(tmp1.getIdx());
    const Ymm ymm_tmp(tmp2.getIdx());
    const Xmm xmm_tmp1(tmp2.getIdx());
    const Xmm xmm_tmp2(tmp3.getIdx());

    if (number_of_values_to_reduce == number_of_f32_in_zmm_) {
        reduce_zmm_to_ymm(zmm_acc, ymm_tmp);
        reduce_ymm_to_xmm(ymm_acc, xmm_tmp1);
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp1);
    } else if (number_of_values_to_reduce > number_of_f32_in_ymm_) {
        vextractf64x4(ymm_acc_upper_half, zmm_acc, 1);
        reduce_ymm_to_scalar(ymm_acc, xmm_tmp1, xmm_tmp2);
        reduce_ymm_to_scalar(ymm_acc_upper_half, xmm_tmp1, xmm_tmp2,
                number_of_values_to_reduce - number_of_f32_in_ymm_);
        vaddps(xmm_acc, xmm_acc, xmm_acc_upper_half);
    } else if (number_of_values_to_reduce <= number_of_f32_in_ymm_) {
        reduce_ymm_to_scalar(
                ymm_acc, xmm_tmp1, xmm_tmp2, number_of_values_to_reduce);
    }
}

template <cpu_isa_t isa>
void jit_rms_kernel<isa>::generate() {
    this->preamble();
    mov(reg_src, ptr[abi_param1 + GET_OFF(src)]);
    mov(reg_scale, ptr[abi_param1 + GET_OFF(scale)]);
    mov(reg_dst, ptr[abi_param1 + GET_OFF(dst)]);
    uni_vpxor(vmm_sum0, vmm_sum0, vmm_sum0);
    uni_vpxor(vmm_sum1, vmm_sum1, vmm_sum1);
    uni_vpxor(vmm_sum2, vmm_sum2, vmm_sum2);
    uni_vpxor(vmm_sum3, vmm_sum3, vmm_sum3);
    mov(reg_src_org, reg_src);

    mov(reg_size, m_jcp.data_size / (vec_size * 4));
    // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
    // sum(x^2)
    align(16);
    if ((m_jcp.data_size / (vec_size * 4)) != 0) {
        Xbyak::Label loop_4reg;
        L(loop_4reg);
        {
            load(vmm_src, reg_src, m_jcp.src_prc, vec_size, false);
            vfmadd231ps(vmm_sum0, vmm_src, vmm_src);
            load(vmm_src, reg_src, m_jcp.src_prc, vec_size, false, vec_size * m_jcp.src_prc.size() * 1);
            vfmadd231ps(vmm_sum1, vmm_src, vmm_src);
            load(vmm_src, reg_src, m_jcp.src_prc, vec_size, false, vec_size * m_jcp.src_prc.size() * 2);
            vfmadd231ps(vmm_sum2, vmm_src, vmm_src);
            load(vmm_src, reg_src, m_jcp.src_prc, vec_size, false, vec_size * m_jcp.src_prc.size() * 3);
            vfmadd231ps(vmm_sum3, vmm_src, vmm_src);

            add(reg_src, vec_size * m_jcp.src_prc.size() * 4);
            dec(reg_size);
            jnz(loop_4reg);
        }
    }

    // 1 ~ 3 vmm
    for (size_t i = m_jcp.data_size / (vec_size * 4) * 4; i < m_jcp.data_size / vec_size; i++) {
        load(vmm_src, reg_src, m_jcp.src_prc, vec_size, false);
        vfmadd231ps(vmm_sum0, vmm_src, vmm_src);
        add(reg_src, vec_size * m_jcp.src_prc.size());
    }
    // tail
    if (m_jcp.data_size % vec_size) {
        load(vmm_src, reg_src, m_jcp.src_prc, m_jcp.data_size % vec_size, false);
        vfmadd231ps(vmm_sum0, vmm_src, vmm_src);
    }
    vaddps(vmm_sum0, vmm_sum0, vmm_sum1);
    vaddps(vmm_sum2, vmm_sum2, vmm_sum3);
    vaddps(vmm_rsqrt, vmm_sum0, vmm_sum2);
    reduce_vmm_to_scalar(vmm_rsqrt, vmm_sum0, vmm_sum1, vmm_sum3, vec_size);

    // mean(x^2)
    mov(reg_tmp.cvt32(), float2int(1.0f / m_jcp.data_size));
    vmovd(xmm_tmp, reg_tmp.cvt32());
    vmulss(xmm_rsqrt, xmm_rsqrt, xmm_tmp);
    // mean(x^2)+eps
    mov(reg_tmp.cvt32(), float2int(m_jcp.eps));
    vmovd(xmm_tmp, reg_tmp.cvt32());
    vaddss(xmm_rsqrt, xmm_rsqrt, xmm_tmp);
    // 1 / sqrt(mean(x^2)+eps) dont's use VRSQRTSS. VRSQRTSS uses approximation and has accuracy issue
    vsqrtss(xmm_rsqrt, xmm_rsqrt, xmm_rsqrt);

    mov(reg_tmp.cvt32(), float2int(1.0f));
    vmovd(xmm_tmp, reg_tmp.cvt32());
    vdivss(xmm_rsqrt, xmm_tmp, xmm_rsqrt);

    // x * rsqrt(mean(x^2)+eps)
    if (m_jcp.scale_size == 1) {
        // rsqrt(mean(x^2)+eps)
        vmovd(xmm_tmp, ptr[reg_scale]);
        vmulss(xmm_rsqrt, xmm_rsqrt, xmm_tmp);
    }
    vbroadcastss(vmm_rsqrt, xmm_rsqrt);
    mov(reg_size, m_jcp.data_size / vec_size);
    mov(reg_src, reg_src_org);
    align(16);
    Xbyak::Label loop_mul;
    L(loop_mul);
    {
        load(vmm_src, reg_src, m_jcp.src_prc, vec_size, false);
        vmulps(vmm_src, vmm_src, vmm_rsqrt);
        if (m_jcp.scale_size != 1) {
            load(vmm_tmp, reg_scale, ov::element::f32, vec_size, false);
            vmulps(vmm_src, vmm_src, vmm_tmp);
        }
        store(reg_dst, vmm_src, m_jcp.dst_prc, vec_size);

        add(reg_src, vec_size * m_jcp.src_prc.size());
        if (m_jcp.scale_size != 1) {
            add(reg_scale, vec_size * sizeof(float));
        }
        add(reg_dst, vec_size * m_jcp.dst_prc.size());
        dec(reg_size);
        jnz(loop_mul);
    }
    // tail
    if (m_jcp.data_size % vec_size) {
        load(vmm_src, reg_src, m_jcp.src_prc, m_jcp.data_size % vec_size, false);
        vmulps(vmm_src, vmm_src, vmm_rsqrt);
        if (m_jcp.scale_size != 1) {
            load(vmm_tmp, reg_scale, ov::element::f32, m_jcp.data_size % vec_size, false);
            vmulps(vmm_src, vmm_src, vmm_tmp);
        }
        store(reg_dst, vmm_src, m_jcp.dst_prc, m_jcp.data_size % vec_size);
    }

    this->postamble();
    for (const auto& emitter : emitters) {
        if (emitter.second)
            emitter.second->emit_data();
    }
}

template <cpu_isa_t isa>
void jit_rms_kernel<isa>::load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, ov::element::Type src_prc, const int& elt_num, bool fill, size_t offset) {
    const auto seed = load_emitter_params(src_prc, ov::element::f32, elt_num, fill, "float_min").hash();
    if (!emitters[seed]) {
        emitters[seed].reset(new jit_load_emitter(this, isa, src_prc, ov::element::f32, elt_num, ov::element::f32, fill, "float_min"));
    }
    emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), offset}, {static_cast<size_t>(vmm_dst.getIdx())},
                                pool_aux_vmm_idxs, pool_aux_gpr_idxs);
}

template <cpu_isa_t isa>
void jit_rms_kernel<isa>::store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, ov::element::Type dst_prc, const int& elt_num, size_t offset) {
    const auto seed = store_emitter_params(ov::element::f32, dst_prc, elt_num).hash();
    if (!emitters[seed]) {
        emitters[seed].reset(new jit_store_emitter(this, isa, ov::element::f32, dst_prc, elt_num));
    }
    emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(reg_dst.getIdx()), offset},
                                pool_aux_vmm_idxs, pool_aux_gpr_idxs);
}

template struct jit_rms_kernel<cpu_isa_t::avx512_core>;
template struct jit_rms_kernel<cpu_isa_t::avx2>;

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov
