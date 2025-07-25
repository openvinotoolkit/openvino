// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_online_softmax_kernel.hpp"

#include <xbyak/xbyak.h>

#include <cassert>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>

#include "openvino/core/type/element_type.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;

#define GET_OFF(field) offsetof(jit_args_online_softmax, field)

namespace ov::intel_cpu::x64 {

template <cpu::x64::cpu_isa_t isa>
void jit_uni_online_softmax_kernel_f32<isa>::generate() {
        load_emitter_vector =
            std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, ov::element::f32, vector_step));
        load_emitter_scalar =
            std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, ov::element::f32, 1));
        store_emitter_vector =
            std::unique_ptr<jit_store_emitter>(new jit_store_emitter(this, isa, ov::element::f32, jcp_.dst_prc, vector_step));
        store_emitter_scalar =
            std::unique_ptr<jit_store_emitter>(new jit_store_emitter(this, isa, ov::element::f32, jcp_.dst_prc, 1));
        load_pool_gpr_idxs = {static_cast<size_t>(reg_aux1.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_aux1.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};
        exp_injector.reset(new jit_uni_eltwise_injector_t<isa>(this,
                                                               dnnl::impl::alg_kind::eltwise_exp,
                                                               0.F,
                                                               0.F,
                                                               1.0F,
                                                               data_type::f32));

        this->preamble();

        mov(reg_data, ptr[reg_params + GET_OFF(data)]);
        mov(reg_max_past, ptr[reg_params + GET_OFF(max_past)]);
        mov(reg_denominator_past, ptr[reg_params + GET_OFF(denominator_past)]);
        mov(reg_max, ptr[reg_params + GET_OFF(max)]);
        mov(reg_denominator, ptr[reg_params + GET_OFF(denominator)]);
        mov(reg_out, ptr[reg_params + GET_OFF(out)]);
        mov(reg_work_amount_outer, ptr[reg_params + GET_OFF(work_amount_outer)]);
        mov(reg_work_amount_inner, ptr[reg_params + GET_OFF(work_amount_inner)]);
        mov(reg_work_amount_inner_head_size, ptr[reg_params + GET_OFF(work_amount_inner_head_size)]);
        mov(reg_table, l_table_constant);

        Xbyak::Label outer_loop_label;
        Xbyak::Label outer_loop_end_label;
        L(outer_loop_label);
        {
            cmp(reg_work_amount_outer, 0);
            jle(outer_loop_end_label, T_NEAR);

            Xbyak::Label max_loop_label;
            Xbyak::Label max_loop_end_label;
            mov(reg_work_amount_inner_aux, reg_work_amount_inner);
            mov(reg_data_aux, reg_data);
            uni_vmovups(vmm_max, ptr[reg_table]);
            L(max_loop_label);
            {
                cmp(reg_work_amount_inner_aux, vector_step);
                jl(max_loop_end_label, T_NEAR);

                load_emitter_vector->emit_code({static_cast<size_t>(reg_data_aux.getIdx())},
                                               {static_cast<size_t>(vmm_val.getIdx())},
                                               {},
                                               {load_pool_gpr_idxs});
                uni_vmaxps(vmm_max, vmm_max, vmm_val);

                add(reg_data_aux, vector_step * sizeof(float));
                sub(reg_work_amount_inner_aux, vector_step);
                jmp(max_loop_label, T_NEAR);
            }
            L(max_loop_end_label);

            // local max
            reduce_vmm(vmm_max.getIdx(), true);
            // update max
            if (jcp_.with_calibration) {
                uni_vmovss(vmm_max_past, ptr[reg_max_past]);
                uni_vmaxps(vmm_max, vmm_max, vmm_max_past);
            }
            uni_vmovss(ptr[reg_max], vmm_max);

            uni_vbroadcastss(vmm_max, Xbyak::Xmm(vmm_max.getIdx()));
            uni_vpxor(vmm_denominator, vmm_denominator, vmm_denominator);
            Xbyak::Label exp_loop_label;
            Xbyak::Label exp_loop_end_label;
            mov(reg_work_amount_inner_aux, reg_work_amount_inner);
            mov(reg_data_aux, reg_data);
            L(exp_loop_label);
            {
                cmp(reg_work_amount_inner_aux, vector_step);
                jl(exp_loop_end_label, T_NEAR);

                load_emitter_vector->emit_code({static_cast<size_t>(reg_data_aux.getIdx())},
                                               {static_cast<size_t>(vmm_val.getIdx())},
                                               {},
                                               {load_pool_gpr_idxs});
                uni_vsubps(vmm_val, vmm_val, vmm_max);
                exp_injector->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
                // store
                store_emitter_vector->emit_code({static_cast<size_t>(vmm_val.getIdx())},
                                                {static_cast<size_t>(reg_data_aux.getIdx())},
                                                {store_pool_vec_idxs},
                                                {store_pool_gpr_idxs});
                uni_vaddps(vmm_denominator, vmm_denominator, vmm_val);

                add(reg_data_aux, vector_step * sizeof(float));
                sub(reg_work_amount_inner_aux, vector_step);
                jmp(exp_loop_label, T_NEAR);
            }
            L(exp_loop_end_label);

            // local denominator
            reduce_vmm(vmm_denominator.getIdx(), false);
            // update denominator
            if (jcp_.with_calibration) {
                uni_vmovss(vmm_denominator_past, ptr[reg_denominator_past]);
                uni_vsubps(vmm_aux1, vmm_max_past, vmm_max);
                exp_injector->compute_vector_range(vmm_aux1.getIdx(), vmm_aux1.getIdx() + 1);
                uni_vmulps(vmm_denominator_past, vmm_denominator_past, vmm_aux1);
                uni_vaddps(vmm_denominator, vmm_denominator, vmm_denominator_past);
            }
            uni_vmovss(ptr[reg_denominator], vmm_denominator);

            uni_vbroadcastss(vmm_denominator, Xbyak::Xmm(vmm_denominator.getIdx()));
            // uni_vdivps(vmm_denominator, 1.0f, vmm_denominator);  // mul in loop if div
            Xbyak::Label div_loop_label;
            Xbyak::Label div_loop_end_label;
            mov(reg_work_amount_inner_aux, reg_work_amount_inner);
            mov(reg_data_aux, reg_data);
            L(div_loop_label);
            {
                cmp(reg_work_amount_inner_aux, vector_step);
                jl(div_loop_end_label, T_NEAR);

                load_emitter_vector->emit_code({static_cast<size_t>(reg_data_aux.getIdx())},
                                               {static_cast<size_t>(vmm_val.getIdx())},
                                               {},
                                               {load_pool_gpr_idxs});
                uni_vdivps(vmm_val, vmm_val, vmm_denominator);
                store_emitter_vector->emit_code({static_cast<size_t>(vmm_val.getIdx())},
                                                {static_cast<size_t>(reg_data_aux.getIdx())},
                                                {store_pool_vec_idxs},
                                                {store_pool_gpr_idxs});

                add(reg_data_aux, vector_step * sizeof(float));
                sub(reg_work_amount_inner_aux, vector_step);
                jmp(div_loop_label, T_NEAR);
            }
            L(div_loop_end_label);

            if (jcp_.with_calibration) {
                // output calibration
                uni_vmovss(vmm_max_past, ptr[reg_max_past]);
                uni_vmovss(vmm_max, ptr[reg_max]);
                uni_vsubps(vmm_max_past, vmm_max_past, vmm_max);
                exp_injector->compute_vector_range(vmm_max_past.getIdx(), vmm_max_past.getIdx() + 1);
                uni_vmovss(vmm_denominator_past, ptr[reg_denominator_past]);
                uni_vmulps(vmm_max_past, vmm_max_past, vmm_denominator_past);
                uni_vmovss(vmm_denominator, ptr[reg_denominator]);
                uni_vdivps(vmm_max_past, vmm_max_past, vmm_denominator);
                uni_vbroadcastss(vmm_max_past, Xbyak::Xmm(vmm_max_past.getIdx()));

                Xbyak::Label calibration_loop_label;
                Xbyak::Label calibration_loop_end_label;
                mov(reg_work_amount_inner_aux, reg_work_amount_inner_head_size);
                mov(reg_data_aux, reg_out);
                L(calibration_loop_label);
                {
                    cmp(reg_work_amount_inner_aux, vector_step);
                    jl(calibration_loop_end_label, T_NEAR);

                    load_emitter_vector->emit_code({static_cast<size_t>(reg_data_aux.getIdx())},
                                                   {static_cast<size_t>(vmm_val.getIdx())},
                                                   {},
                                                   {load_pool_gpr_idxs});
                    uni_vmulps(vmm_val, vmm_val, vmm_max_past);
                    store_emitter_vector->emit_code({static_cast<size_t>(vmm_val.getIdx())},
                                                    {static_cast<size_t>(reg_data_aux.getIdx())},
                                                    {store_pool_vec_idxs},
                                                    {store_pool_gpr_idxs});

                    add(reg_data_aux, vector_step * sizeof(float));
                    sub(reg_work_amount_inner_aux, vector_step);
                    jmp(calibration_loop_label, T_NEAR);
                }
                L(calibration_loop_end_label);
            }
            // update current to past
            uni_vmovss(vmm_max, ptr[reg_max]);
            uni_vmovss(ptr[reg_max_past], vmm_max);
            uni_vmovss(vmm_denominator, ptr[reg_denominator]);
            uni_vmovss(ptr[reg_denominator_past], vmm_denominator);

            mov(reg_aux1, sizeof(float));
            imul(reg_aux1, reg_work_amount_inner);
            add(reg_data, reg_aux1);
            mov(reg_aux1, sizeof(float));
            imul(reg_aux1, reg_work_amount_inner_head_size);
            add(reg_out, reg_aux1);
            add(reg_max_past, sizeof(float));
            add(reg_denominator_past, sizeof(float));
            add(reg_max, sizeof(float));
            add(reg_denominator, sizeof(float));

            sub(reg_work_amount_outer, 1);
            jmp(outer_loop_label, T_NEAR);
        }
        L(outer_loop_end_label);

        this->postamble();

        load_emitter_vector->emit_data();
        load_emitter_scalar->emit_data();
        store_emitter_vector->emit_data();
        store_emitter_scalar->emit_data();

        prepare_table();

        exp_injector->prepare_table();
    }

// kernel has if? if yes need separate function
template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_uni_online_softmax_kernel_f32<isa>::reduce_xmm(Xbyak::Xmm xmm_val, bool is_max) {
    // val:4,3,2,1 -> aux3:4,4,2,2
    uni_vmovshdup(xmm_aux3, xmm_val);
    // val:4v4, 3v4, 2v2, 1v2
    if (is_max) {
        uni_vmaxps(xmm_val, xmm_val, xmm_aux3);
    } else {
        uni_vaddps(xmm_val, xmm_val, xmm_aux3);
    }
    // aux3:4,4,4v4,3v4
    uni_vmovhlps(xmm_aux3, xmm_aux3, xmm_val);
    // ......,1v2v3v4
    if (is_max) {
        uni_vmaxps(xmm_val, xmm_val, xmm_aux3);
    } else {
        uni_vaddps(xmm_val, xmm_val, xmm_aux3);
    }
}
// reduce to lowerest of vmm_idx register
template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_uni_online_softmax_kernel_f32<isa>::reduce_vmm(int vmm_idx, bool is_max) {
    if (isa == cpu::x64::avx2) {
        auto ymm_val = Xbyak::Ymm(vmm_idx);
        auto xmm_val = Xbyak::Xmm(vmm_idx);
        vextractf128(xmm_aux1, ymm_val, 0);
        vextractf128(xmm_aux2, ymm_val, 1);
        if (is_max) {
            uni_vmaxps(xmm_val, xmm_aux1, xmm_aux2);
        } else {
            uni_vaddps(xmm_val, xmm_aux1, xmm_aux2);
        }
        reduce_xmm(xmm_val, is_max);
    } else if (isa == cpu::x64::avx512_core) {
        auto zmm_val = Xbyak::Zmm(vmm_idx);
        auto xmm_val = Xbyak::Xmm(vmm_idx);
        vextractf32x4(xmm_aux1, zmm_val, 0);
        vextractf32x4(xmm_aux2, zmm_val, 1);
        if (is_max) {
            uni_vmaxps(xmm_aux1, xmm_aux1, xmm_aux2);
        } else {
            uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
        }
        vextractf32x4(xmm_aux2, zmm_val, 2);
        vextractf32x4(xmm_aux3, zmm_val, 3);
        if (is_max) {
            uni_vmaxps(xmm_aux2, xmm_aux2, xmm_aux3);
            uni_vmaxps(xmm_val, xmm_aux1, xmm_aux2);
        } else {
            uni_vaddps(xmm_aux2, xmm_aux2, xmm_aux3);
            uni_vaddps(xmm_val, xmm_aux1, xmm_aux2);
        }
        reduce_xmm(xmm_val, is_max);
    }
}

template struct jit_uni_online_softmax_kernel_f32<cpu::x64::avx2>;
template struct jit_uni_online_softmax_kernel_f32<cpu::x64::avx512_core>;

}  // namespace ov::intel_cpu::x64