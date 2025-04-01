// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_max_suppression.hpp"

#include <memory>

#include "utils/general_utils.h"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;

#define GET_OFF(field) offsetof(NmsCallArgs, field)

namespace ov::intel_cpu::kernel {

template <x64::cpu_isa_t isa>
void NonMaxSuppression<isa>::generate() {
    load_vector_emitter =
        std::make_unique<jit_load_emitter>(this, isa, ov::element::f32, ov::element::f32, vector_step);
    load_scalar_emitter =
        std::make_unique<jit_load_emitter>(this, isa, ov::element::f32, ov::element::f32, scalar_step);

    exp_injector.reset(
        new x64::jit_uni_eltwise_injector<isa>(this, dnnl::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.f, data_type::f32));

    this->preamble();

    uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

    load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()),
                          static_cast<size_t>(reg_load_table.getIdx())};
    store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
    store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

    mov(reg_boxes_coord0, ptr[reg_params + GET_OFF(selected_boxes_coord[0])]);
    mov(reg_boxes_coord1, ptr[reg_params + GET_OFF(selected_boxes_coord[0]) + 1 * sizeof(size_t)]);
    mov(reg_boxes_coord2, ptr[reg_params + GET_OFF(selected_boxes_coord[0]) + 2 * sizeof(size_t)]);
    mov(reg_boxes_coord3, ptr[reg_params + GET_OFF(selected_boxes_coord[0]) + 3 * sizeof(size_t)]);
    mov(reg_candidate_box, ptr[reg_params + GET_OFF(candidate_box)]);
    mov(reg_candidate_status, ptr[reg_params + GET_OFF(candidate_status)]);
    mov(reg_boxes_num, ptr[reg_params + GET_OFF(selected_boxes_num)]);
    mov(reg_iou_threshold, ptr[reg_params + GET_OFF(iou_threshold)]);
    // soft
    mov(reg_score_threshold, ptr[reg_params + GET_OFF(score_threshold)]);
    mov(reg_score, ptr[reg_params + GET_OFF(score)]);
    mov(reg_scale, ptr[reg_params + GET_OFF(scale)]);

    // could use rcx(reg_table) and rdi(reg_temp) now as abi parse finished
    mov(reg_table, l_table_constant);
    if (x64::mayiuse(x64::avx512_core)) {
        kmovw(k_mask_one, word[reg_table + vlen]);
    }
    uni_vbroadcastss(vmm_iou_threshold, ptr[reg_iou_threshold]);
    uni_vbroadcastss(vmm_score_threshold, ptr[reg_score_threshold]);

    uni_vbroadcastss(vmm_candidate_coord0, ptr[reg_candidate_box]);
    uni_vbroadcastss(vmm_candidate_coord1, ptr[reg_candidate_box + 1 * sizeof(float)]);
    uni_vbroadcastss(vmm_candidate_coord2, ptr[reg_candidate_box + 2 * sizeof(float)]);
    uni_vbroadcastss(vmm_candidate_coord3, ptr[reg_candidate_box + 3 * sizeof(float)]);

    if (m_jcp.box_encode_type == NMSBoxEncodeType::CORNER) {
        // box format: y1, x1, y2, x2
        uni_vminps(vmm_temp1, vmm_candidate_coord0, vmm_candidate_coord2);
        uni_vmaxps(vmm_temp2, vmm_candidate_coord0, vmm_candidate_coord2);
        uni_vmovups(vmm_candidate_coord0, vmm_temp1);
        uni_vmovups(vmm_candidate_coord2, vmm_temp2);

        uni_vminps(vmm_temp1, vmm_candidate_coord1, vmm_candidate_coord3);
        uni_vmaxps(vmm_temp2, vmm_candidate_coord1, vmm_candidate_coord3);
        uni_vmovups(vmm_candidate_coord1, vmm_temp1);
        uni_vmovups(vmm_candidate_coord3, vmm_temp2);
    } else {
        // box format: x_center, y_center, width, height --> y1, x1, y2, x2
        uni_vmulps(vmm_temp1, vmm_candidate_coord2, ptr[reg_table]);  // width/2
        uni_vmulps(vmm_temp2, vmm_candidate_coord3, ptr[reg_table]);  // height/2

        uni_vaddps(vmm_temp3, vmm_candidate_coord0, vmm_temp1);  // x_center + width/2
        uni_vmovups(vmm_candidate_coord3, vmm_temp3);

        uni_vaddps(vmm_temp3, vmm_candidate_coord1, vmm_temp2);  // y_center + height/2
        uni_vmovups(vmm_candidate_coord2, vmm_temp3);

        uni_vsubps(vmm_temp3, vmm_candidate_coord0, vmm_temp1);  // x_center - width/2
        uni_vsubps(vmm_temp4, vmm_candidate_coord1, vmm_temp2);  // y_center - height/2

        uni_vmovups(vmm_candidate_coord1, vmm_temp3);
        uni_vmovups(vmm_candidate_coord0, vmm_temp4);
    }

    // check from last to first
    imul(reg_temp_64, reg_boxes_num, sizeof(float));
    add(reg_boxes_coord0, reg_temp_64);  // y1
    add(reg_boxes_coord1, reg_temp_64);  // x1
    add(reg_boxes_coord2, reg_temp_64);  // y2
    add(reg_boxes_coord3, reg_temp_64);  // x2

    Xbyak::Label hard_nms_label;
    Xbyak::Label nms_end_label;

    mov(reg_temp_32, ptr[reg_scale]);
    test(reg_temp_32, reg_temp_32);
    jz(hard_nms_label, T_NEAR);

    soft_nms();

    jmp(nms_end_label, T_NEAR);

    L(hard_nms_label);

    hard_nms();

    L(nms_end_label);

    this->postamble();

    load_vector_emitter->emit_data();
    load_scalar_emitter->emit_data();

    prepare_table();
    exp_injector->prepare_table();
}

template <x64::cpu_isa_t isa>
void NonMaxSuppression<isa>::hard_nms() {
    Xbyak::Label main_loop_label_hard;
    Xbyak::Label main_loop_end_label_hard;
    Xbyak::Label tail_loop_label_hard;
    Xbyak::Label terminate_label_hard;
    L(main_loop_label_hard);
    {
        cmp(reg_boxes_num, vector_step);
        jl(main_loop_end_label_hard, T_NEAR);

        sub(reg_boxes_coord0, vector_step * sizeof(float));
        sub(reg_boxes_coord1, vector_step * sizeof(float));
        sub(reg_boxes_coord2, vector_step * sizeof(float));
        sub(reg_boxes_coord3, vector_step * sizeof(float));

        // iou result is in vmm_temp3
        iou(vector_step);

        sub(reg_boxes_num, vector_step);

        suppressed_by_iou(false);

        // if zero continue, else set result to suppressed and terminate
        jz(main_loop_label_hard, T_NEAR);

        uni_vpextrd(ptr[reg_candidate_status], Xbyak::Xmm(vmm_zero.getIdx()), 0);

        jmp(terminate_label_hard, T_NEAR);
    }
    L(main_loop_end_label_hard);

    L(tail_loop_label_hard);
    {
        cmp(reg_boxes_num, 1);
        jl(terminate_label_hard, T_NEAR);

        sub(reg_boxes_coord0, scalar_step * sizeof(float));
        sub(reg_boxes_coord1, scalar_step * sizeof(float));
        sub(reg_boxes_coord2, scalar_step * sizeof(float));
        sub(reg_boxes_coord3, scalar_step * sizeof(float));

        // iou result is in vmm_temp3
        iou(scalar_step);

        sub(reg_boxes_num, scalar_step);

        suppressed_by_iou(true);

        jz(tail_loop_label_hard, T_NEAR);

        uni_vpextrd(ptr[reg_candidate_status], Xbyak::Xmm(vmm_zero.getIdx()), 0);

        jmp(terminate_label_hard, T_NEAR);
    }

    L(terminate_label_hard);
}

template <x64::cpu_isa_t isa>
void NonMaxSuppression<isa>::soft_nms() {
    uni_vbroadcastss(vmm_scale, ptr[reg_scale]);

    Xbyak::Label main_loop_label;
    Xbyak::Label main_loop_end_label;
    Xbyak::Label tail_loop_label;
    Xbyak::Label terminate_label;

    Xbyak::Label main_loop_label_soft;
    Xbyak::Label tail_loop_label_soft;
    L(main_loop_label);
    {
        cmp(reg_boxes_num, vector_step);
        jl(main_loop_end_label, T_NEAR);

        sub(reg_boxes_coord0, vector_step * sizeof(float));
        sub(reg_boxes_coord1, vector_step * sizeof(float));
        sub(reg_boxes_coord2, vector_step * sizeof(float));
        sub(reg_boxes_coord3, vector_step * sizeof(float));

        // result(iou and weight) is in vmm_temp3
        iou(vector_step);
        sub(reg_boxes_num, vector_step);

        // soft suppressed by iou_threshold
        if (m_jcp.is_soft_suppressed_by_iou) {
            suppressed_by_iou(false);

            // if zero continue soft suppression, else set result to suppressed and terminate
            jz(main_loop_label_soft, T_NEAR);

            uni_vpextrd(ptr[reg_candidate_status], Xbyak::Xmm(vmm_zero.getIdx()), 0);

            jmp(terminate_label, T_NEAR);

            L(main_loop_label_soft);
        }

        // weight: std::exp(scale * iou * iou)
        soft_coeff();

        // vector weights multiply
        horizontal_mul();

        uni_vbroadcastss(vmm_temp1, ptr[reg_score]);

        // new score in vmm3[0]
        uni_vmulps(vmm_temp3, vmm_temp3, vmm_temp1);
        // store new score
        uni_vmovss(ptr[reg_score], vmm_temp3);

        // cmpps(_CMP_LE_OS) if new score is less or equal than score_threshold
        suppressed_by_score();

        jz(main_loop_label, T_NEAR);

        uni_vpextrd(ptr[reg_candidate_status], Xbyak::Xmm(vmm_zero.getIdx()), 0);

        jmp(terminate_label, T_NEAR);
    }
    L(main_loop_end_label);

    L(tail_loop_label);
    {
        cmp(reg_boxes_num, 1);
        jl(terminate_label, T_NEAR);

        sub(reg_boxes_coord0, scalar_step * sizeof(float));
        sub(reg_boxes_coord1, scalar_step * sizeof(float));
        sub(reg_boxes_coord2, scalar_step * sizeof(float));
        sub(reg_boxes_coord3, scalar_step * sizeof(float));

        iou(scalar_step);
        sub(reg_boxes_num, scalar_step);

        // soft suppressed by iou_threshold
        if (m_jcp.is_soft_suppressed_by_iou) {
            suppressed_by_iou(true);

            jz(tail_loop_label_soft, T_NEAR);

            uni_vpextrd(ptr[reg_candidate_status], Xbyak::Xmm(vmm_zero.getIdx()), 0);

            jmp(terminate_label, T_NEAR);

            L(tail_loop_label_soft);
        }

        soft_coeff();

        uni_vbroadcastss(vmm_temp1, ptr[reg_score]);

        // vmm3[0] is valide, no need horizontal mul.
        uni_vmulps(vmm_temp3, vmm_temp3, vmm_temp1);

        uni_vmovss(ptr[reg_score], vmm_temp3);

        // cmpps(_CMP_LE_OS) if new score is less or equal than score_threshold
        suppressed_by_score();

        jz(tail_loop_label, T_NEAR);

        uni_vpextrd(ptr[reg_candidate_status], Xbyak::Xmm(vmm_zero.getIdx()), 0);

        jmp(terminate_label, T_NEAR);
    }

    L(terminate_label);
}

template <x64::cpu_isa_t isa>
void NonMaxSuppression<isa>::suppressed_by_iou(bool is_scalar) {
    if (x64::mayiuse(x64::avx512_core)) {
        vcmpps(k_mask, vmm_temp3, vmm_iou_threshold, 0x0D);  // _CMP_GE_OS. vcmpps w/ kmask only on V5
        if (is_scalar) {
            kandw(k_mask, k_mask, k_mask_one);
        }
        kortestw(k_mask, k_mask);  // bitwise check if all zero
    } else if (x64::mayiuse(x64::avx)) {
        // vex instructions with xmm on avx and ymm on avx2
        vcmpps(vmm_temp4, vmm_temp3, vmm_iou_threshold, 0x0D);  // xmm and ymm only on V1.
        if (is_scalar) {
            uni_vpextrd(reg_temp_32, Xbyak::Xmm(vmm_temp4.getIdx()), 0);
            test(reg_temp_32, reg_temp_32);
        } else {
            uni_vtestps(vmm_temp4,
                        vmm_temp4);  // vtestps: sign bit check if all zeros, ymm and xmm only on V1, N/A on V5
        }
    } else {
        // pure sse path, make sure don't spoil vmm_temp3, which may used in after soft-suppression
        uni_vmovups(vmm_temp4, vmm_temp3);
        cmpps(vmm_temp4, vmm_iou_threshold, 0x07);  // order compare, 0 for at least one is NaN

        uni_vmovups(vmm_temp2, vmm_temp3);
        cmpps(vmm_temp2, vmm_iou_threshold, 0x05);  // _CMP_GE_US on sse, no direct _CMP_GE_OS supported.

        uni_vandps(vmm_temp4, vmm_temp4, vmm_temp2);
        if (is_scalar) {
            uni_vpextrd(reg_temp_32, Xbyak::Xmm(vmm_temp4.getIdx()), 0);
            test(reg_temp_32, reg_temp_32);
        } else {
            uni_vtestps(vmm_temp4, vmm_temp4);  // ptest: bitwise check if all zeros, on sse41
        }
    }
}

template <x64::cpu_isa_t isa>
void NonMaxSuppression<isa>::suppressed_by_score() {
    if (x64::mayiuse(x64::avx512_core)) {
        vcmpps(k_mask,
               vmm_temp3,
               vmm_score_threshold,
               0x02);  // vcmpps w/ kmask only on V5, w/o kmask version N/A on V5
        kandw(k_mask, k_mask, k_mask_one);
        kortestw(k_mask, k_mask);  // bitwise check if all zero
    } else if (x64::mayiuse(x64::avx)) {
        vcmpps(vmm_temp4, vmm_temp3, vmm_score_threshold, 0x02);
        uni_vpextrd(reg_temp_32, Xbyak::Xmm(vmm_temp4.getIdx()), 0);
        test(reg_temp_32, reg_temp_32);
    } else {
        cmpps(vmm_temp3, vmm_score_threshold, 0x02);  // _CMP_LE_OS on sse
        uni_vpextrd(reg_temp_32, Xbyak::Xmm(vmm_temp3.getIdx()), 0);
        test(reg_temp_32, reg_temp_32);
    }
}

template <x64::cpu_isa_t isa>
void NonMaxSuppression<isa>::iou(int ele_num) {
    auto load = [&](Xbyak::Reg64 reg_src, Vmm vmm_dst) {
        if (ele_num != scalar_step && ele_num != vector_step) {
            OPENVINO_THROW("NMS JIT implementation supports load emitter with only element count scalar_step or "
                           "vector_step! Get: ",
                           ele_num);
        }

        const auto& load_emitter = ele_num == 1 ? load_scalar_emitter : load_vector_emitter;
        load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())},
                                {static_cast<size_t>(vmm_dst.getIdx())},
                                {},
                                {load_pool_gpr_idxs});
    };
    load(reg_boxes_coord0, vmm_boxes_coord0);
    load(reg_boxes_coord1, vmm_boxes_coord1);
    load(reg_boxes_coord2, vmm_boxes_coord2);
    load(reg_boxes_coord3, vmm_boxes_coord3);

    if (m_jcp.box_encode_type == NMSBoxEncodeType::CORNER) {
        // box format: y1, x1, y2, x2
        uni_vminps(vmm_temp1, vmm_boxes_coord0, vmm_boxes_coord2);
        uni_vmaxps(vmm_temp2, vmm_boxes_coord0, vmm_boxes_coord2);
        uni_vmovups(vmm_boxes_coord0, vmm_temp1);
        uni_vmovups(vmm_boxes_coord2, vmm_temp2);

        uni_vminps(vmm_temp1, vmm_boxes_coord1, vmm_boxes_coord3);
        uni_vmaxps(vmm_temp2, vmm_boxes_coord1, vmm_boxes_coord3);
        uni_vmovups(vmm_boxes_coord1, vmm_temp1);
        uni_vmovups(vmm_boxes_coord3, vmm_temp2);
    } else {
        // box format: x_center, y_center, width, height --> y1, x1, y2, x2
        uni_vmulps(vmm_temp1, vmm_boxes_coord2, ptr[reg_table]);  // width/2
        uni_vmulps(vmm_temp2, vmm_boxes_coord3, ptr[reg_table]);  // height/2

        uni_vaddps(vmm_temp3, vmm_boxes_coord0, vmm_temp1);  // x_center + width/2
        uni_vmovups(vmm_boxes_coord3, vmm_temp3);

        uni_vaddps(vmm_temp3, vmm_boxes_coord1, vmm_temp2);  // y_center + height/2
        uni_vmovups(vmm_boxes_coord2, vmm_temp3);

        uni_vsubps(vmm_temp3, vmm_boxes_coord0, vmm_temp1);  // x_center - width/2
        uni_vsubps(vmm_temp4, vmm_boxes_coord1, vmm_temp2);  // y_center - height/2

        uni_vmovups(vmm_boxes_coord1, vmm_temp3);
        uni_vmovups(vmm_boxes_coord0, vmm_temp4);
    }

    uni_vsubps(vmm_temp1, vmm_boxes_coord2, vmm_boxes_coord0);
    uni_vsubps(vmm_temp2, vmm_boxes_coord3, vmm_boxes_coord1);
    uni_vmulps(vmm_temp1, vmm_temp1, vmm_temp2);  // boxes area

    uni_vsubps(vmm_temp2, vmm_candidate_coord2, vmm_candidate_coord0);
    uni_vsubps(vmm_temp3, vmm_candidate_coord3, vmm_candidate_coord1);
    uni_vmulps(vmm_temp2, vmm_temp2, vmm_temp3);  // candidate(bc) area  // candidate area calculate once and check if 0

    uni_vaddps(vmm_temp1, vmm_temp1, vmm_temp2);  // areaI + areaJ to free vmm_temp2

    // y of intersection
    uni_vminps(vmm_temp3, vmm_boxes_coord2, vmm_candidate_coord2);  // min(Ymax)
    uni_vmaxps(vmm_temp4, vmm_boxes_coord0, vmm_candidate_coord0);  // max(Ymin)
    uni_vsubps(vmm_temp3, vmm_temp3, vmm_temp4);                    // min(Ymax) - max(Ymin)
    uni_vmaxps(vmm_temp3, vmm_temp3, vmm_zero);

    // x of intersection
    uni_vminps(vmm_temp4, vmm_boxes_coord3, vmm_candidate_coord3);  // min(Xmax)
    uni_vmaxps(vmm_temp2, vmm_boxes_coord1, vmm_candidate_coord1);  // max(Xmin)
    uni_vsubps(vmm_temp4, vmm_temp4, vmm_temp2);                    // min(Xmax) - max(Xmin)
    uni_vmaxps(vmm_temp4, vmm_temp4, vmm_zero);

    // intersection_area
    uni_vmulps(vmm_temp3, vmm_temp3, vmm_temp4);

    // iou: intersection_area / (areaI + areaJ - intersection_area);
    uni_vsubps(vmm_temp1, vmm_temp1, vmm_temp3);
    uni_vdivps(vmm_temp3, vmm_temp3, vmm_temp1);
}

// std::exp(scale * iou * iou)
template <x64::cpu_isa_t isa>
void NonMaxSuppression<isa>::soft_coeff() {
    uni_vmulps(vmm_temp3, vmm_temp3, vmm_temp3);
    uni_vmulps(vmm_temp3, vmm_temp3, vmm_scale);
    exp_injector->compute_vector_range(vmm_temp3.getIdx(), vmm_temp3.getIdx() + 1);
}

template <x64::cpu_isa_t isa>
void NonMaxSuppression<isa>::horizontal_mul_xmm(const Xbyak::Xmm& xmm_weight, const Xbyak::Xmm& xmm_aux) {
    uni_vmovshdup(xmm_aux, xmm_weight);           //  weight:1,2,3,4; aux:2,2,4,4
    uni_vmulps(xmm_weight, xmm_weight, xmm_aux);  //  weight:1*2,2*2,3*4,4*4
    uni_vmovhlps(xmm_aux, xmm_aux, xmm_weight);   //  aux:3*4,4*4,4,4
    uni_vmulps(xmm_weight, xmm_weight, xmm_aux);  //  weight:1*2*3*4,...
}

// horizontal mul for vmm_weight(Vmm(3)), temp1 and temp2 as aux
template <x64::cpu_isa_t isa>
inline void NonMaxSuppression<isa>::horizontal_mul() {
    auto xmm_weight = Xbyak::Xmm(vmm_temp3.getIdx());
    auto xmm_temp1 = Xbyak::Xmm(vmm_temp1.getIdx());
    auto xmm_temp2 = Xbyak::Xmm(vmm_temp2.getIdx());
    if (isa == x64::sse41) {
        horizontal_mul_xmm(xmm_weight, xmm_temp1);
    } else if (isa == x64::avx2) {
        auto ymm_weight = Xbyak::Ymm(vmm_temp3.getIdx());
        vextractf128(xmm_temp1, ymm_weight, 0);
        vextractf128(xmm_temp2, ymm_weight, 1);
        uni_vmulps(xmm_weight, xmm_temp1, xmm_temp2);
        horizontal_mul_xmm(xmm_weight, xmm_temp1);
    } else {
        auto zmm_weight = Xbyak::Zmm(vmm_temp3.getIdx());
        vextractf32x4(xmm_temp1, zmm_weight, 0);
        vextractf32x4(xmm_temp2, zmm_weight, 1);
        uni_vmulps(xmm_temp1, xmm_temp1, xmm_temp2);
        vextractf32x4(xmm_temp2, zmm_weight, 2);
        vextractf32x4(xmm_weight, zmm_weight, 3);
        uni_vmulps(xmm_weight, xmm_weight, xmm_temp2);
        uni_vmulps(xmm_weight, xmm_weight, xmm_temp1);
        horizontal_mul_xmm(xmm_weight, xmm_temp1);
    }
}

template class NonMaxSuppression<x64::avx512_core>;
template class NonMaxSuppression<x64::avx2>;
template class NonMaxSuppression<x64::sse41>;

}  // namespace ov::intel_cpu::kernel
