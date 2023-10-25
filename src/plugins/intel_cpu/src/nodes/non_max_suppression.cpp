// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <queue>

#include "non_max_suppression.h"
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset5.hpp>
#include <ov_ops/nms_ie_internal.hpp>
#include "utils/general_utils.h"

#include "cpu/x64/jit_generator.hpp"
#include "emitters/x64/jit_load_store_emitters.hpp"
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include <shape_inference/shape_inference_internal_dyn.hpp>

using namespace InferenceEngine;
using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_nms_args, field)

namespace ov {
namespace intel_cpu {
namespace node {

#if defined(OPENVINO_ARCH_X86_64)
template <cpu_isa_t isa>
struct jit_uni_nms_kernel_f32 : public jit_uni_nms_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_nms_kernel_f32)

    explicit jit_uni_nms_kernel_f32(jit_nms_config_params jcp_) : jit_uni_nms_kernel(jcp_), jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        load_vector_emitter.reset(new jit_load_emitter(this, isa, Precision::FP32, Precision::FP32, vector_step));
        load_scalar_emitter.reset(new jit_load_emitter(this, isa, Precision::FP32, Precision::FP32, scalar_step));

        exp_injector.reset(new jit_uni_eltwise_injector_f32<isa>(this, dnnl::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.0f));

        this->preamble();

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
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
        if (mayiuse(cpu::x64::avx512_core)) {
            kmovw(k_mask_one, word[reg_table + vlen]);
        }
        uni_vbroadcastss(vmm_iou_threshold, ptr[reg_iou_threshold]);
        uni_vbroadcastss(vmm_score_threshold, ptr[reg_score_threshold]);

        uni_vbroadcastss(vmm_candidate_coord0, ptr[reg_candidate_box]);
        uni_vbroadcastss(vmm_candidate_coord1, ptr[reg_candidate_box + 1 * sizeof(float)]);
        uni_vbroadcastss(vmm_candidate_coord2, ptr[reg_candidate_box + 2 * sizeof(float)]);
        uni_vbroadcastss(vmm_candidate_coord3, ptr[reg_candidate_box + 3 * sizeof(float)]);

        if (jcp.box_encode_type == NMSBoxEncodeType::CORNER) {
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
            uni_vmulps(vmm_temp1, vmm_candidate_coord2, ptr[reg_table]);   // width/2
            uni_vmulps(vmm_temp2, vmm_candidate_coord3, ptr[reg_table]);   // height/2

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

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;
    const int vector_step = vlen / sizeof(float);
    const int scalar_step = 1;

    Xbyak::Reg64 reg_boxes_coord0 = r8;
    Xbyak::Reg64 reg_boxes_coord1 = r9;
    Xbyak::Reg64 reg_boxes_coord2 = r10;
    Xbyak::Reg64 reg_boxes_coord3 = r11;
    Xbyak::Reg64 reg_candidate_box = r12;
    Xbyak::Reg64 reg_candidate_status = r13;
    Xbyak::Reg64 reg_boxes_num = r14;
    Xbyak::Reg64 reg_iou_threshold = r15;
    // more for soft
    Xbyak::Reg64 reg_score_threshold = rdx;
    Xbyak::Reg64 reg_score = rbp;
    Xbyak::Reg64 reg_scale = rsi;

    Xbyak::Reg64 reg_load_table = rax;
    Xbyak::Reg64 reg_load_store_mask = rbx;

    // reuse
    Xbyak::Label l_table_constant;
    Xbyak::Reg64 reg_table = rcx;
    Xbyak::Reg64 reg_temp_64 = rdi;
    Xbyak::Reg32 reg_temp_32 = edi;

    Xbyak::Reg64 reg_params = abi_param1;

    std::unique_ptr<jit_load_emitter> load_vector_emitter = nullptr;
    std::unique_ptr<jit_load_emitter> load_scalar_emitter = nullptr;

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;
    std::vector<size_t> load_pool_gpr_idxs;

    Vmm vmm_boxes_coord0 = Vmm(1);
    Vmm vmm_boxes_coord1 = Vmm(2);
    Vmm vmm_boxes_coord2 = Vmm(3);
    Vmm vmm_boxes_coord3 = Vmm(4);
    Vmm vmm_candidate_coord0 = Vmm(5);
    Vmm vmm_candidate_coord1 = Vmm(6);
    Vmm vmm_candidate_coord2 = Vmm(7);
    Vmm vmm_candidate_coord3 = Vmm(8);
    Vmm vmm_temp1 = Vmm(9);
    Vmm vmm_temp2 = Vmm(10);
    Vmm vmm_temp3 = Vmm(11);
    Vmm vmm_temp4 = Vmm(12);

    Vmm vmm_iou_threshold = Vmm(13);
    Vmm vmm_zero = Vmm(15);

    // soft
    Vmm vmm_score_threshold = Vmm(14);
    Vmm vmm_scale = Vmm(0);

    Xbyak::Opmask k_mask = Xbyak::Opmask(7);
    Xbyak::Opmask k_mask_one = Xbyak::Opmask(6);

    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector;

    inline void hard_nms() {
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

            uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

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

            uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

            jmp(terminate_label_hard, T_NEAR);
        }

        L(terminate_label_hard);
    }

    inline void soft_nms() {
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
            if (jcp.is_soft_suppressed_by_iou) {
                suppressed_by_iou(false);

                // if zero continue soft suppression, else set result to suppressed and terminate
                jz(main_loop_label_soft, T_NEAR);

                uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

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

            uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

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
            if (jcp.is_soft_suppressed_by_iou) {
                suppressed_by_iou(true);

                jz(tail_loop_label_soft, T_NEAR);

                uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

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

            uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

            jmp(terminate_label, T_NEAR);
        }

        L(terminate_label);
    }

    inline void suppressed_by_iou(bool is_scalar) {
        if (mayiuse(cpu::x64::avx512_core)) {
            vcmpps(k_mask, vmm_temp3, vmm_iou_threshold, 0x0D); // _CMP_GE_OS. vcmpps w/ kmask only on V5
            if (is_scalar)
                kandw(k_mask, k_mask, k_mask_one);
            kortestw(k_mask, k_mask);    // bitwise check if all zero
        } else if (mayiuse(cpu::x64::avx)) {
            // vex instructions with xmm on avx and ymm on avx2
            vcmpps(vmm_temp4, vmm_temp3, vmm_iou_threshold, 0x0D);  // xmm and ymm only on V1.
            if (is_scalar) {
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);
            } else {
                uni_vtestps(vmm_temp4, vmm_temp4);  // vtestps: sign bit check if all zeros, ymm and xmm only on V1, N/A on V5
            }
        } else {
            // pure sse path, make sure don't spoil vmm_temp3, which may used in after soft-suppression
            uni_vmovups(vmm_temp4, vmm_temp3);
            cmpps(vmm_temp4, vmm_iou_threshold, 0x07);  // order compare, 0 for at least one is NaN

            uni_vmovups(vmm_temp2, vmm_temp3);
            cmpps(vmm_temp2, vmm_iou_threshold, 0x05);   // _CMP_GE_US on sse, no direct _CMP_GE_OS supported.

            uni_vandps(vmm_temp4, vmm_temp4, vmm_temp2);
            if (is_scalar) {
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);
            } else {
                uni_vtestps(vmm_temp4, vmm_temp4);  // ptest: bitwise check if all zeros, on sse41
            }
        }
    }

    inline void suppressed_by_score() {
        if (mayiuse(cpu::x64::avx512_core)) {
            vcmpps(k_mask, vmm_temp3, vmm_score_threshold, 0x02); // vcmpps w/ kmask only on V5, w/o kmask version N/A on V5
            kandw(k_mask, k_mask, k_mask_one);
            kortestw(k_mask, k_mask);    // bitwise check if all zero
        } else if (mayiuse(cpu::x64::avx)) {
            vcmpps(vmm_temp4, vmm_temp3, vmm_score_threshold, 0x02);
            uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
            test(reg_temp_32, reg_temp_32);
        } else {
            cmpps(vmm_temp3, vmm_score_threshold, 0x02);  // _CMP_LE_OS on sse
            uni_vpextrd(reg_temp_32, Xmm(vmm_temp3.getIdx()), 0);
            test(reg_temp_32, reg_temp_32);
        }
    }

    inline void iou(int ele_num) {
        auto load = [&](Xbyak::Reg64 reg_src, Vmm vmm_dst) {
            if (ele_num != scalar_step && ele_num != vector_step)
                IE_THROW() << "NMS JIT implementation supports load emitter with only element count scalar_step or vector_step! Get: " << ele_num;

            const auto& load_emitter = ele_num == 1 ? load_scalar_emitter : load_vector_emitter;
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())},
                {}, {load_pool_gpr_idxs});
        };
        load(reg_boxes_coord0, vmm_boxes_coord0);
        load(reg_boxes_coord1, vmm_boxes_coord1);
        load(reg_boxes_coord2, vmm_boxes_coord2);
        load(reg_boxes_coord3, vmm_boxes_coord3);

        if (jcp.box_encode_type == NMSBoxEncodeType::CORNER) {
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
            uni_vmulps(vmm_temp1, vmm_boxes_coord2, ptr[reg_table]);   // width/2
            uni_vmulps(vmm_temp2, vmm_boxes_coord3, ptr[reg_table]);   // height/2

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
        uni_vsubps(vmm_temp3, vmm_temp3, vmm_temp4);  // min(Ymax) - max(Ymin)
        uni_vmaxps(vmm_temp3, vmm_temp3, vmm_zero);

        // x of intersection
        uni_vminps(vmm_temp4, vmm_boxes_coord3, vmm_candidate_coord3);  // min(Xmax)
        uni_vmaxps(vmm_temp2, vmm_boxes_coord1, vmm_candidate_coord1);  // max(Xmin)
        uni_vsubps(vmm_temp4, vmm_temp4, vmm_temp2);  // min(Xmax) - max(Xmin)
        uni_vmaxps(vmm_temp4, vmm_temp4, vmm_zero);

        // intersection_area
        uni_vmulps(vmm_temp3, vmm_temp3, vmm_temp4);

        // iou: intersection_area / (areaI + areaJ - intersection_area);
        uni_vsubps(vmm_temp1, vmm_temp1, vmm_temp3);
        uni_vdivps(vmm_temp3, vmm_temp3, vmm_temp1);
    }

    // std::exp(scale * iou * iou)
    inline void soft_coeff() {
        uni_vmulps(vmm_temp3, vmm_temp3, vmm_temp3);
        uni_vmulps(vmm_temp3, vmm_temp3, vmm_scale);
        exp_injector->compute_vector_range(vmm_temp3.getIdx(), vmm_temp3.getIdx() + 1);
    }

    inline void horizontal_mul_xmm(const Xbyak::Xmm &xmm_weight, const Xbyak::Xmm &xmm_aux) {
        uni_vmovshdup(xmm_aux, xmm_weight);              //  weight:1,2,3,4; aux:2,2,4,4
        uni_vmulps(xmm_weight, xmm_weight, xmm_aux);     //  weight:1*2,2*2,3*4,4*4
        uni_vmovhlps(xmm_aux, xmm_aux, xmm_weight);      //  aux:3*4,4*4,4,4
        uni_vmulps(xmm_weight, xmm_weight, xmm_aux);     //  weight:1*2*3*4,...
    }

    // horizontal mul for vmm_weight(Vmm(3)), temp1 and temp2 as aux
    inline void horizontal_mul() {
        Xbyak::Xmm xmm_weight = Xbyak::Xmm(vmm_temp3.getIdx());
        Xbyak::Xmm xmm_temp1 = Xbyak::Xmm(vmm_temp1.getIdx());
        Xbyak::Xmm xmm_temp2 = Xbyak::Xmm(vmm_temp2.getIdx());
        if (isa == cpu::x64::sse41) {
            horizontal_mul_xmm(xmm_weight, xmm_temp1);
        } else if (isa == cpu::x64::avx2) {
            Xbyak::Ymm ymm_weight = Xbyak::Ymm(vmm_temp3.getIdx());
            vextractf128(xmm_temp1, ymm_weight, 0);
            vextractf128(xmm_temp2, ymm_weight, 1);
            uni_vmulps(xmm_weight, xmm_temp1, xmm_temp2);
            horizontal_mul_xmm(xmm_weight, xmm_temp1);
        } else {
            Xbyak::Zmm zmm_weight = Xbyak::Zmm(vmm_temp3.getIdx());
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

    inline void prepare_table() {
        auto broadcast_d = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(int); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table_constant);
        broadcast_d(0x3f000000);   // 0.5f
        dw(0x0001);
    }
};
#endif

bool NonMaxSuppression::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        using NonMaxSuppressionV9 = ngraph::op::v9::NonMaxSuppression;
        if (!one_of(op->get_type_info(), NonMaxSuppressionV9::get_type_info_static(),
                    ov::op::internal::NonMaxSuppressionIEInternal::get_type_info_static())) {
            errorMessage = "Only NonMaxSuppression v9 and NonMaxSuppressionIEInternal are supported";
            return false;
        }

        if (const auto nms9 = std::dynamic_pointer_cast<const NonMaxSuppressionV9>(op)) {
            const auto boxEncoding = nms9->get_box_encoding();
            if (!one_of(boxEncoding, NonMaxSuppressionV9::BoxEncodingType::CENTER, NonMaxSuppressionV9::BoxEncodingType::CORNER)) {
                errorMessage = "Supports only CENTER and CORNER box encoding type";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

NonMaxSuppression::NonMaxSuppression(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()),
      isSoftSuppressedByIOU(false) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "NMS layer with name '" + op->get_friendly_name() + "' ";
    if (one_of(op->get_type_info(), ov::op::internal::NonMaxSuppressionIEInternal::get_type_info_static()))
        m_outStaticShape = true;

    if (getOriginalInputsNumber() < 2 || getOriginalInputsNumber() > 6)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getOriginalInputsNumber();

    if (getOriginalOutputsNumber() != 3)
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getOriginalOutputsNumber();

    if (const auto nms9 = std::dynamic_pointer_cast<const ngraph::op::v9::NonMaxSuppression>(op)) {
        boxEncodingType = static_cast<NMSBoxEncodeType>(nms9->get_box_encoding());
        sortResultDescending = nms9->get_sort_result_descending();
        } else if (const auto nmsIe = std::dynamic_pointer_cast<const ov::op::internal::NonMaxSuppressionIEInternal>(op)) {
            boxEncodingType = nmsIe->m_center_point_box ? NMSBoxEncodeType::CENTER : NMSBoxEncodeType::CORNER;
            sortResultDescending = nmsIe->m_sort_result_descending;
        } else {
            const auto &typeInfo = op->get_type_info();
            IE_THROW() << errorPrefix << " doesn't support NMS: " << typeInfo.name << " v" << typeInfo.version_id;
        }

        const auto &boxes_dims = getInputShapeAtPort(NMS_BOXES).getDims();
        if (boxes_dims.size() != 3)
            IE_THROW() << errorPrefix << "has unsupported 'boxes' input rank: " << boxes_dims.size();
        if (boxes_dims[2] != 4)
            IE_THROW() << errorPrefix << "has unsupported 'boxes' input 3rd dimension size: " << boxes_dims[2];

        const auto &scores_dims = getInputShapeAtPort(NMS_SCORES).getDims();
        if (scores_dims.size() != 3)
            IE_THROW() << errorPrefix << "has unsupported 'scores' input rank: " << scores_dims.size();

        const Shape valid_outputs_shape = getOutputShapeAtPort(NMS_VALIDOUTPUTS);
        if (valid_outputs_shape.getRank() != 1)
            IE_THROW() << errorPrefix << "has unsupported 'valid_outputs' output rank: " << valid_outputs_shape.getRank();
        if (valid_outputs_shape.getDims()[0] != 1)
            IE_THROW() << errorPrefix << "has unsupported 'valid_outputs' output 1st dimension size: " << valid_outputs_shape.getDims()[1];
}

void NonMaxSuppression::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32, Precision::BF16, Precision::FP16};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", inType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_VALIDOUTPUTS), supportedIntOutputPrecision, "valid_outputs", outType);

    const std::vector<Precision> supportedPrecision = {Precision::I16, Precision::U8, Precision::I8, Precision::U16, Precision::I32,
                                                       Precision::U32, Precision::I64, Precision::U64};

    if (inputShapes.size() > NMS_MAXOUTPUTBOXESPERCLASS)
        check1DInput(getInputShapeAtPort(NMS_MAXOUTPUTBOXESPERCLASS), supportedPrecision, "max_output_boxes_per_class", NMS_MAXOUTPUTBOXESPERCLASS);
    if (inputShapes.size() > NMS_IOUTHRESHOLD)
        check1DInput(getInputShapeAtPort(NMS_IOUTHRESHOLD), supportedFloatPrecision, "iou_threshold", NMS_IOUTHRESHOLD);
    if (inputShapes.size() > NMS_SCORETHRESHOLD)
        check1DInput(getInputShapeAtPort(NMS_SCORETHRESHOLD), supportedFloatPrecision, "score_threshold", NMS_SCORETHRESHOLD);
    if (inputShapes.size() > NMS_SOFTNMSSIGMA)
        check1DInput(getInputShapeAtPort(NMS_SCORETHRESHOLD), supportedFloatPrecision, "soft_nms_sigma", NMS_SCORETHRESHOLD);

    checkOutput(getOutputShapeAtPort(NMS_SELECTEDINDICES), supportedIntOutputPrecision, "selected_indices", NMS_SELECTEDINDICES);
    checkOutput(getOutputShapeAtPort(NMS_SELECTEDSCORES), supportedFloatPrecision, "selected_scores", NMS_SELECTEDSCORES);

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        Precision inPrecision = i == NMS_MAXOUTPUTBOXESPERCLASS ? Precision::I32 : Precision::FP32;
        inDataConf.emplace_back(LayoutType::ncsp, inPrecision);
    }

    std::vector<PortConfigurator> outDataConf;
    outDataConf.reserve(outputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); ++i) {
        Precision outPrecision = i == NMS_SELECTEDSCORES ? Precision::FP32 : Precision::I32;
        outDataConf.emplace_back(LayoutType::ncsp, outPrecision);
    }

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    addSupportedPrimDesc(inDataConf, outDataConf, impl_type);

    // as only FP32 and ncsp is supported, and kernel is shape agnostic, we can create here. There is no need to recompilation.
    createJitKernel();
}

void NonMaxSuppression::prepareParams() {
    const auto& boxesDims = isDynamicNode() ? getParentEdgesAtPort(NMS_BOXES)[0]->getMemory().getStaticDims() :
                                               getInputShapeAtPort(NMS_BOXES).getStaticDims();
    const auto& scoresDims = isDynamicNode() ? getParentEdgesAtPort(NMS_SCORES)[0]->getMemory().getStaticDims() :
                                                getInputShapeAtPort(NMS_SCORES).getStaticDims();

    numBatches = boxesDims[0];
    numBoxes = boxesDims[1];
    numClasses = scoresDims[1];
    if (numBatches != scoresDims[0])
        IE_THROW() << errorPrefix << " numBatches is different in 'boxes' and 'scores' inputs";
    if (numBoxes != scoresDims[2])
        IE_THROW() << errorPrefix << " numBoxes is different in 'boxes' and 'scores' inputs";

    numFiltBox.resize(numBatches);
    for (auto & i : numFiltBox)
        i.resize(numClasses);
}

bool NonMaxSuppression::isExecutable() const {
    return isDynamicNode() || Node::isExecutable();
}

void NonMaxSuppression::createJitKernel() {
#if defined(OPENVINO_ARCH_X86_64)
    auto jcp = jit_nms_config_params();
    jcp.box_encode_type = boxEncodingType;
    jcp.is_soft_suppressed_by_iou = isSoftSuppressedByIOU;

    if (mayiuse(cpu::x64::avx512_core)) {
        nms_kernel.reset(new jit_uni_nms_kernel_f32<cpu::x64::avx512_core>(jcp));
    } else if (mayiuse(cpu::x64::avx2)) {
        nms_kernel.reset(new jit_uni_nms_kernel_f32<cpu::x64::avx2>(jcp));
    } else if (mayiuse(cpu::x64::sse41)) {
        nms_kernel.reset(new jit_uni_nms_kernel_f32<cpu::x64::sse41>(jcp));
    }

    if (nms_kernel)
        nms_kernel->create_ker();
#endif
}

void NonMaxSuppression::executeDynamicImpl(dnnl::stream strm) {
    if (hasEmptyInputTensors() || (inputShapes.size() > NMS_MAXOUTPUTBOXESPERCLASS &&
            reinterpret_cast<int *>(getParentEdgeAt(NMS_MAXOUTPUTBOXESPERCLASS)->getMemoryPtr()->getData())[0] == 0)) {
        redefineOutputMemory({{0, 3}, {0, 3}, {1}});
        *reinterpret_cast<int *>(getChildEdgesAtPort(NMS_VALIDOUTPUTS)[0]->getMemoryPtr()->getData()) = 0;
        return;
    }
    execute(strm);
}

void NonMaxSuppression::execute(dnnl::stream strm) {
    const float *boxes = reinterpret_cast<const float *>(getParentEdgeAt(NMS_BOXES)->getMemoryPtr()->getData());
    const float *scores = reinterpret_cast<const float *>(getParentEdgeAt(NMS_SCORES)->getMemoryPtr()->getData());

    if (inputShapes.size() > NMS_MAXOUTPUTBOXESPERCLASS) {
        maxOutputBoxesPerClass = reinterpret_cast<int *>(getParentEdgeAt(NMS_MAXOUTPUTBOXESPERCLASS)->getMemoryPtr()->getData())[0];
    }

    maxOutputBoxesPerClass = std::min(maxOutputBoxesPerClass, numBoxes);

    if (maxOutputBoxesPerClass == 0) {
        return;
    }

    if (inputShapes.size() > NMS_IOUTHRESHOLD)
        iouThreshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_IOUTHRESHOLD)->getMemoryPtr()->getData())[0];

    if (inputShapes.size() > NMS_SCORETHRESHOLD)
        scoreThreshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_SCORETHRESHOLD)->getMemoryPtr()->getData())[0];

    if (inputShapes.size() > NMS_SOFTNMSSIGMA)
        softNMSSigma = reinterpret_cast<float *>(getParentEdgeAt(NMS_SOFTNMSSIGMA)->getMemoryPtr()->getData())[0];
    scale = 0.0f;
    if (softNMSSigma > 0.0) {
        scale = -0.5f / softNMSSigma;
    }

    auto boxesStrides = getParentEdgeAt(NMS_BOXES)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();
    auto scoresStrides = getParentEdgeAt(NMS_SCORES)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();

    const auto maxNumberOfBoxes = maxOutputBoxesPerClass * numBatches * numClasses;
    std::vector<filteredBoxes> filtBoxes(maxNumberOfBoxes);

    if (softNMSSigma == 0.0f) {
        nmsWithoutSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
    } else {
        nmsWithSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
    }

    size_t startOffset = numFiltBox[0][0];
    for (size_t b = 0; b < numFiltBox.size(); b++) {
        size_t batchOffset = b*numClasses*maxOutputBoxesPerClass;
        for (size_t c = (b == 0 ? 1 : 0); c < numFiltBox[b].size(); c++) {
            size_t offset = batchOffset + c*maxOutputBoxesPerClass;
            for (size_t i = 0; i < numFiltBox[b][c]; i++) {
                filtBoxes[startOffset + i] = filtBoxes[offset + i];
            }
            startOffset += numFiltBox[b][c];
        }
    }
    filtBoxes.resize(startOffset);

    // need more particular comparator to get deterministic behaviour
    // escape situation when filtred boxes with same score have different position from launch to launch
    if (sortResultDescending) {
        parallel_sort(filtBoxes.begin(), filtBoxes.end(),
                      [](const filteredBoxes& l, const filteredBoxes& r) {
                          return (l.score > r.score) ||
                                 (l.score ==  r.score && l.batch_index < r.batch_index) ||
                                 (l.score ==  r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                                 (l.score ==  r.score && l.batch_index == r.batch_index && l.class_index == r.class_index && l.box_index < r.box_index);
                      });
    }

    auto indicesMemPtr = getChildEdgesAtPort(NMS_SELECTEDINDICES)[0]->getMemoryPtr();
    auto scoresMemPtr =  getChildEdgesAtPort(NMS_SELECTEDSCORES)[0]->getMemoryPtr();
    const size_t validOutputs = std::min(filtBoxes.size(), maxNumberOfBoxes);

    if (!m_outStaticShape) {
        VectorDims newDims{validOutputs, 3};
        redefineOutputMemory({newDims, newDims, {1}});
    }

    int selectedIndicesStride = indicesMemPtr->getDescWithType<BlockedMemoryDesc>()->getStrides()[0];

    int *selectedIndicesPtr = reinterpret_cast<int *>(indicesMemPtr->getData());
    float *selectedScoresPtr = reinterpret_cast<float *>(scoresMemPtr->getData());

    size_t idx = 0lu;
    for (; idx < validOutputs; idx++) {
        selectedIndicesPtr[0] = filtBoxes[idx].batch_index;
        selectedIndicesPtr[1] = filtBoxes[idx].class_index;
        selectedIndicesPtr[2] = filtBoxes[idx].box_index;
        selectedIndicesPtr += selectedIndicesStride;

        selectedScoresPtr[0] = static_cast<float>(filtBoxes[idx].batch_index);
        selectedScoresPtr[1] = static_cast<float>(filtBoxes[idx].class_index);
        selectedScoresPtr[2] = static_cast<float>(filtBoxes[idx].score);
        selectedScoresPtr += selectedIndicesStride;
    }

    if (m_outStaticShape) {
        std::fill(selectedIndicesPtr, selectedIndicesPtr + (maxNumberOfBoxes - idx) * selectedIndicesStride, -1);
        std::fill(selectedScoresPtr, selectedScoresPtr + (maxNumberOfBoxes - idx) * selectedIndicesStride, -1.f);
    }

    int *valid_outputs = reinterpret_cast<int *>(getChildEdgesAtPort(NMS_VALIDOUTPUTS)[0]->getMemoryPtr()->getData());
    *valid_outputs = static_cast<int>(validOutputs);
}

bool NonMaxSuppression::created() const {
    return getType() == Type::NonMaxSuppression;
}

float NonMaxSuppression::intersectionOverUnion(const float *boxesI, const float *boxesJ) {
    float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
    if (boxEncodingType == NMSBoxEncodeType::CENTER) {
        //  box format: x_center, y_center, width, height
        yminI = boxesI[1] - boxesI[3] / 2.f;
        xminI = boxesI[0] - boxesI[2] / 2.f;
        ymaxI = boxesI[1] + boxesI[3] / 2.f;
        xmaxI = boxesI[0] + boxesI[2] / 2.f;
        yminJ = boxesJ[1] - boxesJ[3] / 2.f;
        xminJ = boxesJ[0] - boxesJ[2] / 2.f;
        ymaxJ = boxesJ[1] + boxesJ[3] / 2.f;
        xmaxJ = boxesJ[0] + boxesJ[2] / 2.f;
    } else {
        //  box format: y1, x1, y2, x2
        yminI = (std::min)(boxesI[0], boxesI[2]);
        xminI = (std::min)(boxesI[1], boxesI[3]);
        ymaxI = (std::max)(boxesI[0], boxesI[2]);
        xmaxI = (std::max)(boxesI[1], boxesI[3]);
        yminJ = (std::min)(boxesJ[0], boxesJ[2]);
        xminJ = (std::min)(boxesJ[1], boxesJ[3]);
        ymaxJ = (std::max)(boxesJ[0], boxesJ[2]);
        xmaxJ = (std::max)(boxesJ[1], boxesJ[3]);
    }

    float areaI = (ymaxI - yminI) * (xmaxI - xminI);
    float areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
    if (areaI <= 0.f || areaJ <= 0.f)
        return 0.f;

    float intersection_area =
            (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ), 0.f) *
            (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ), 0.f);
    return intersection_area / (areaI + areaJ - intersection_area);
}

void NonMaxSuppression::nmsWithSoftSigma(const float *boxes, const float *scores, const VectorDims &boxesStrides,
                                                             const VectorDims &scoresStrides, std::vector<filteredBoxes> &filtBoxes) {
    auto less = [](const boxInfo& l, const boxInfo& r) {
        return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
    };

    // update score, if iou is 0, weight is 1, score does not change
    // if is_soft_suppressed_by_iou is false, apply for all iou, including iou>iou_threshold, soft suppressed when score < score_threshold
    // if is_soft_suppressed_by_iou is true, hard suppressed by iou_threshold, then soft suppress
    auto coeff = [&](float iou) {
        if (isSoftSuppressedByIOU && iou > iouThreshold)
            return 0.0f;
        return std::exp(scale * iou * iou);
    };

    parallel_for2d(numBatches, numClasses, [&](int batch_idx, int class_idx) {
        std::vector<filteredBoxes> selectedBoxes;
        const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(less);  // score, box_id, suppress_begin_index
        for (int box_idx = 0; box_idx < static_cast<int>(numBoxes); box_idx++) {
            if (scoresPtr[box_idx] > scoreThreshold)
                sorted_boxes.emplace(boxInfo({scoresPtr[box_idx], box_idx, 0}));
        }
        size_t sortedBoxSize = sorted_boxes.size();
        size_t maxSeletedBoxNum = std::min(sortedBoxSize, maxOutputBoxesPerClass);
        selectedBoxes.reserve(maxSeletedBoxNum);
        if (maxSeletedBoxNum > 0) {
            // include first directly
            boxInfo candidateBox = sorted_boxes.top();
            sorted_boxes.pop();
            selectedBoxes.push_back({ candidateBox.score, batch_idx, class_idx, candidateBox.idx });
            if (maxSeletedBoxNum > 1) {
                if (nms_kernel) {
                    std::vector<float> boxCoord0(maxSeletedBoxNum, 0.0f);
                    std::vector<float> boxCoord1(maxSeletedBoxNum, 0.0f);
                    std::vector<float> boxCoord2(maxSeletedBoxNum, 0.0f);
                    std::vector<float> boxCoord3(maxSeletedBoxNum, 0.0f);

                    boxCoord0[0] = boxesPtr[candidateBox.idx * 4];
                    boxCoord1[0] = boxesPtr[candidateBox.idx * 4 + 1];
                    boxCoord2[0] = boxesPtr[candidateBox.idx * 4 + 2];
                    boxCoord3[0] = boxesPtr[candidateBox.idx * 4 + 3];

                    auto arg = jit_nms_args();
                    arg.iou_threshold = static_cast<float*>(&iouThreshold);
                    arg.score_threshold = static_cast<float*>(&scoreThreshold);
                    arg.scale = static_cast<float*>(&scale);
                    while (selectedBoxes.size() < maxOutputBoxesPerClass && !sorted_boxes.empty()) {
                        boxInfo candidateBox = sorted_boxes.top();
                        float origScore = candidateBox.score;
                        sorted_boxes.pop();

                        int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected, 2 for updated
                        arg.score = static_cast<float*>(&candidateBox.score);
                        arg.selected_boxes_num = selectedBoxes.size() - candidateBox.suppress_begin_index;
                        arg.selected_boxes_coord[0] = static_cast<float*>(&boxCoord0[candidateBox.suppress_begin_index]);
                        arg.selected_boxes_coord[1] = static_cast<float*>(&boxCoord1[candidateBox.suppress_begin_index]);
                        arg.selected_boxes_coord[2] = static_cast<float*>(&boxCoord2[candidateBox.suppress_begin_index]);
                        arg.selected_boxes_coord[3] = static_cast<float*>(&boxCoord3[candidateBox.suppress_begin_index]);
                        arg.candidate_box = static_cast<const float*>(&boxesPtr[candidateBox.idx * 4]);
                        arg.candidate_status = static_cast<int*>(&candidateStatus);
                        (*nms_kernel)(&arg);

                        if (candidateStatus == NMSCandidateStatus::SUPPRESSED) {
                            continue;
                        } else {
                            if (candidateBox.score == origScore) {
                                selectedBoxes.push_back({ candidateBox.score, batch_idx, class_idx, candidateBox.idx });
                                int selectedSize = selectedBoxes.size();
                                boxCoord0[selectedSize - 1] = boxesPtr[candidateBox.idx * 4];
                                boxCoord1[selectedSize - 1] = boxesPtr[candidateBox.idx * 4 + 1];
                                boxCoord2[selectedSize - 1] = boxesPtr[candidateBox.idx * 4 + 2];
                                boxCoord3[selectedSize - 1] = boxesPtr[candidateBox.idx * 4 + 3];
                            } else {
                                candidateBox.suppress_begin_index = selectedBoxes.size();
                                sorted_boxes.push(candidateBox);
                            }
                        }
                    }
                } else {
                    while (selectedBoxes.size() < maxOutputBoxesPerClass && !sorted_boxes.empty()) {
                        boxInfo candidateBox = sorted_boxes.top();
                        float origScore = candidateBox.score;
                        sorted_boxes.pop();

                        int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected, 2 for updated
                        for (int selected_idx = static_cast<int>(selectedBoxes.size()) - 1; selected_idx >= candidateBox.suppress_begin_index; selected_idx--) {
                            float iou = intersectionOverUnion(&boxesPtr[candidateBox.idx * 4], &boxesPtr[selectedBoxes[selected_idx].box_index * 4]);

                            // when is_soft_suppressed_by_iou is true, score is decayed to zero and implicitely suppressed if iou > iou_threshold.
                            candidateBox.score *= coeff(iou);
                            // soft suppressed
                            if (candidateBox.score <= scoreThreshold) {
                                candidateStatus = NMSCandidateStatus::SUPPRESSED;
                                break;
                            }
                        }

                        if (candidateStatus == NMSCandidateStatus::SUPPRESSED) {
                            continue;
                        } else {
                            if (candidateBox.score == origScore) {
                                selectedBoxes.push_back({ candidateBox.score, batch_idx, class_idx, candidateBox.idx });
                            } else {
                                candidateBox.suppress_begin_index = selectedBoxes.size();
                                sorted_boxes.push(candidateBox);
                            }
                        }
                    }
                }
            }
        }
        numFiltBox[batch_idx][class_idx] = selectedBoxes.size();
        size_t offset = batch_idx*numClasses*maxOutputBoxesPerClass + class_idx*maxOutputBoxesPerClass;
        for (size_t i = 0; i < selectedBoxes.size(); i++) {
            filtBoxes[offset + i] = selectedBoxes[i];
        }
    });
}

void NonMaxSuppression::nmsWithoutSoftSigma(const float *boxes, const float *scores, const VectorDims &boxesStrides,
                                                                const VectorDims &scoresStrides, std::vector<filteredBoxes> &filtBoxes) {
    int max_out_box = static_cast<int>(maxOutputBoxesPerClass);
    parallel_for2d(numBatches, numClasses, [&](int batch_idx, int class_idx) {
        const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::vector<std::pair<float, int>> sorted_boxes;  // score, box_idx
        for (size_t box_idx = 0; box_idx < numBoxes; box_idx++) {
            if (scoresPtr[box_idx] > scoreThreshold)
                sorted_boxes.emplace_back(std::make_pair(scoresPtr[box_idx], box_idx));
        }

        int io_selection_size = 0;
        size_t sortedBoxSize = sorted_boxes.size();
        if (sortedBoxSize > 0) {
            parallel_sort(sorted_boxes.begin(), sorted_boxes.end(),
                          [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                              return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                          });
            int offset = batch_idx*numClasses*maxOutputBoxesPerClass + class_idx*maxOutputBoxesPerClass;
            filtBoxes[offset + 0] = filteredBoxes(sorted_boxes[0].first, batch_idx, class_idx, sorted_boxes[0].second);
            io_selection_size++;
            if (sortedBoxSize > 1) {
                if (nms_kernel) {
                    std::vector<float> boxCoord0(sortedBoxSize, 0.0f);
                    std::vector<float> boxCoord1(sortedBoxSize, 0.0f);
                    std::vector<float> boxCoord2(sortedBoxSize, 0.0f);
                    std::vector<float> boxCoord3(sortedBoxSize, 0.0f);

                    boxCoord0[0] = boxesPtr[sorted_boxes[0].second * 4];
                    boxCoord1[0] = boxesPtr[sorted_boxes[0].second * 4 + 1];
                    boxCoord2[0] = boxesPtr[sorted_boxes[0].second * 4 + 2];
                    boxCoord3[0] = boxesPtr[sorted_boxes[0].second * 4 + 3];

                    auto arg = jit_nms_args();
                    arg.iou_threshold = static_cast<float*>(&iouThreshold);
                    arg.score_threshold = static_cast<float*>(&scoreThreshold);
                    arg.scale = static_cast<float*>(&scale);
                    // box start index do not change for hard supresion
                    arg.selected_boxes_coord[0] = static_cast<float*>(&boxCoord0[0]);
                    arg.selected_boxes_coord[1] = static_cast<float*>(&boxCoord1[0]);
                    arg.selected_boxes_coord[2] = static_cast<float*>(&boxCoord2[0]);
                    arg.selected_boxes_coord[3] = static_cast<float*>(&boxCoord3[0]);

                    for (size_t candidate_idx = 1; (candidate_idx < sortedBoxSize) && (io_selection_size < max_out_box); candidate_idx++) {
                        int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected
                        arg.selected_boxes_num = io_selection_size;
                        arg.candidate_box = static_cast<const float*>(&boxesPtr[sorted_boxes[candidate_idx].second * 4]);
                        arg.candidate_status = static_cast<int*>(&candidateStatus);
                        (*nms_kernel)(&arg);
                        if (candidateStatus == NMSCandidateStatus::SELECTED) {
                            boxCoord0[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * 4];
                            boxCoord1[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * 4 + 1];
                            boxCoord2[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * 4 + 2];
                            boxCoord3[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * 4 + 3];
                            filtBoxes[offset + io_selection_size] =
                                filteredBoxes(sorted_boxes[candidate_idx].first, batch_idx, class_idx, sorted_boxes[candidate_idx].second);
                            io_selection_size++;
                        }
                    }
                } else {
                    for (size_t candidate_idx = 1; (candidate_idx < sortedBoxSize) && (io_selection_size < max_out_box); candidate_idx++) {
                        int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected
                        for (int selected_idx = io_selection_size - 1; selected_idx >= 0; selected_idx--) {
                            float iou = intersectionOverUnion(&boxesPtr[sorted_boxes[candidate_idx].second * 4],
                                &boxesPtr[filtBoxes[offset + selected_idx].box_index * 4]);
                            if (iou >= iouThreshold) {
                                candidateStatus = NMSCandidateStatus::SUPPRESSED;
                                break;
                            }
                        }

                        if (candidateStatus == NMSCandidateStatus::SELECTED) {
                            filtBoxes[offset + io_selection_size] =
                                filteredBoxes(sorted_boxes[candidate_idx].first, batch_idx, class_idx, sorted_boxes[candidate_idx].second);
                            io_selection_size++;
                        }
                    }
                }
            }
        }

        numFiltBox[batch_idx][class_idx] = io_selection_size;
    });
}

void NonMaxSuppression::checkPrecision(const Precision& prec, const std::vector<Precision>& precList,
                                                           const std::string& name, const std::string& type) {
    if (std::find(precList.begin(), precList.end(), prec) == precList.end())
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' " << type << " precision: " << prec;
}

void NonMaxSuppression::check1DInput(const Shape& shape, const std::vector<Precision>& precList,
                                                         const std::string& name, const size_t port) {
    checkPrecision(getOriginalInputPrecisionAtPort(port), precList, name, inType);

    if (shape.getRank() != 0 && shape.getRank() != 1)
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' input rank: " << shape.getRank();
    if (shape.getRank() == 1)
        if (shape.getDims()[0] != 1)
            IE_THROW() << errorPrefix << "has unsupported '" << name << "' input 1st dimension size: " << MemoryDescUtils::dim2str(shape.getDims()[0]);
}

void NonMaxSuppression::checkOutput(const Shape& shape, const std::vector<Precision>& precList,
                                                        const std::string& name, const size_t port) {
    checkPrecision(getOriginalOutputPrecisionAtPort(port), precList, name, outType);

    if (shape.getRank() != 2)
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' output rank: " << shape.getRank();
    if (shape.getDims()[1] != 3)
        IE_THROW() << errorPrefix << "has unsupported '" << name << "' output 2nd dimension size: " << MemoryDescUtils::dim2str(shape.getDims()[1]);
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
