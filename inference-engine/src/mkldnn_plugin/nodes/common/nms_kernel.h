// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include "emitters/jit_load_store_emitters.hpp"
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>

using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define BOX_COORD_NUM 4

namespace MKLDNNPlugin {

enum class NMSBoxEncodeType {
    CORNER,
    CENTER
};

enum NMSCandidateStatus {
    SUPPRESSED = 0,
    SELECTED = 1,
    UPDATED = 2
};

struct jit_nms_config_params {
    NMSBoxEncodeType box_encode_type;
    bool is_soft_suppressed_by_iou;
};

struct jit_nms_args {
    const void* selected_boxes_coord[BOX_COORD_NUM];
    size_t selected_boxes_num;
    const void* candidate_box;
    const void* iou_threshold;
    void* candidate_status;
    // for soft suppression, score *= scale * iou * iou;
    const void* score_threshold;
    const void* scale;
    void* score;
};

struct jit_uni_nms_kernel {
    void (*ker_)(const jit_nms_args *);

    void operator()(const jit_nms_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_nms_kernel(jit_nms_config_params jcp_) : ker_(nullptr), jcp(jcp_) {}
    virtual ~jit_uni_nms_kernel() {}

    virtual void create_ker() = 0;

    jit_nms_config_params jcp;
};

#define GET_OFF(field) offsetof(jit_nms_args, field)

template <cpu_isa_t isa>
struct jit_uni_nms_kernel_f32 : public jit_uni_nms_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_nms_kernel_f32)

    explicit jit_uni_nms_kernel_f32(jit_nms_config_params jcp_) : jit_uni_nms_kernel(jcp_), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        load_emitter.reset(new jit_load_emitter(this, isa, nullptr));
        store_emitter.reset(new jit_store_emitter(this, isa, nullptr));
        exp_injector.reset(new jit_uni_eltwise_injector_f32<isa>(this, mkldnn::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.0f));

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
        if (mayiuse(cpu::x64::avx512_common)) {
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

        load_emitter->emit_data();
        store_emitter->emit_data();

        prepare_table();
        exp_injector->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

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

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;

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
        int step = vlen / sizeof(float);
        Xbyak::Label main_loop_label_hard;
        Xbyak::Label main_loop_end_label_hard;
        Xbyak::Label tail_loop_label_hard;
        Xbyak::Label terminate_label_hard;
        L(main_loop_label_hard);
        {
            cmp(reg_boxes_num, step);
            jl(main_loop_end_label_hard, T_NEAR);

            sub(reg_boxes_coord0, step * sizeof(float));
            sub(reg_boxes_coord1, step * sizeof(float));
            sub(reg_boxes_coord2, step * sizeof(float));
            sub(reg_boxes_coord3, step * sizeof(float));

            // iou result is in vmm_temp3
            iou(step);

            sub(reg_boxes_num, step);

            if (mayiuse(cpu::x64::avx512_common)) {
                vcmpps(k_mask, vmm_temp3, vmm_iou_threshold, 0x0D); // _CMP_GE_OS. vcmpps w/ kmask only on V5
                kortestw(k_mask, k_mask);    // bitwise check if all zero
            } else if (mayiuse(cpu::x64::avx)) {
                // vex instructions with xmm on avx and ymm on avx2
                vcmpps(vmm_temp4, vmm_temp3, vmm_iou_threshold, 0x0D);  // xmm and ymm only on V1.
                uni_vtestps(vmm_temp4, vmm_temp4);  // vtestps: sign bit check if all zeros, ymm and xmm only on V1, N/A on V5
            } else {
                // pure sse path
                uni_vmovups(vmm_temp4, vmm_temp3);
                cmpps(vmm_temp4, vmm_iou_threshold, 0x07);  // order compare, 0 for unorders

                cmpps(vmm_temp3, vmm_iou_threshold, 0x05);   // _CMP_GE_US on sse, no direct _CMP_GE_OS supported.

                uni_vandps(vmm_temp4, vmm_temp4, vmm_temp3);
                uni_vtestps(vmm_temp4, vmm_temp4);  // ptest: bitwise check if all zeros, on sse41
            }
            // if zero continue, else set result to suppressed and terminate
            jz(main_loop_label_hard, T_NEAR);

            uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

            jmp(terminate_label_hard, T_NEAR);
        }
        L(main_loop_end_label_hard);

        step = 1;
        L(tail_loop_label_hard);
        {
            cmp(reg_boxes_num, 1);
            jl(terminate_label_hard, T_NEAR);

            sub(reg_boxes_coord0, step * sizeof(float));
            sub(reg_boxes_coord1, step * sizeof(float));
            sub(reg_boxes_coord2, step * sizeof(float));
            sub(reg_boxes_coord3, step * sizeof(float));

            // iou result is in vmm_temp3
            iou(step);

            sub(reg_boxes_num, step);

            if (mayiuse(cpu::x64::avx512_common)) {
                vcmpps(k_mask, vmm_temp3, vmm_iou_threshold, 0x0D); // vcmpps w/ kmask only on V5
                kandw(k_mask, k_mask, k_mask_one);
                kortestw(k_mask, k_mask);    // bitwise check if all zero
            } else if (mayiuse(cpu::x64::avx)) {
                // vex instructions with xmm on avx and ymm on avx2
                vcmpps(vmm_temp4, vmm_temp3, vmm_iou_threshold, 0x0D);  // xmm and ymm only on V1.
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);
            } else {
                uni_vmovups(vmm_temp4, vmm_temp3);

                cmpps(vmm_temp4, vmm_iou_threshold, 0x07);  // order compare
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);

                jz(tail_loop_label_hard, T_NEAR);

                cmpps(vmm_temp3, vmm_iou_threshold, 0x05);  // _CMP_GE_US on sse
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp3.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);
            }

            jz(tail_loop_label_hard, T_NEAR);

            uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

            jmp(terminate_label_hard, T_NEAR);
        }

        L(terminate_label_hard);
    }

    inline void soft_nms() {
        uni_vbroadcastss(vmm_scale, ptr[reg_scale]);

        int step = vlen / sizeof(float);
        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label terminate_label;

        Xbyak::Label main_loop_label_soft;
        Xbyak::Label tail_loop_label_soft;
        L(main_loop_label);
        {
            cmp(reg_boxes_num, step);
            jl(main_loop_end_label, T_NEAR);

            sub(reg_boxes_coord0, step * sizeof(float));
            sub(reg_boxes_coord1, step * sizeof(float));
            sub(reg_boxes_coord2, step * sizeof(float));
            sub(reg_boxes_coord3, step * sizeof(float));

            // result(iou and weight) is in vmm_temp3
            iou(step);
            sub(reg_boxes_num, step);

            // soft suppressed by iou_threshold
            if (jcp.is_soft_suppressed_by_iou) {
                if (mayiuse(cpu::x64::avx512_common)) {
                    vcmpps(k_mask, vmm_temp3, vmm_iou_threshold, 0x0D); // _CMP_GE_OS. vcmpps w/ kmask only on V5
                    kortestw(k_mask, k_mask);    // bitwise check if all zero
                } else if (mayiuse(cpu::x64::avx)) {
                    // vex instructions with xmm on avx and ymm on avx2
                    vcmpps(vmm_temp4, vmm_temp3, vmm_iou_threshold, 0x0D);  // xmm and ymm only on V1.
                    uni_vtestps(vmm_temp4, vmm_temp4);  // vtestps: sign bit check if all zeros, ymm and xmm only on V1, N/A on V5
                } else {
                    // pure sse path
                    uni_vmovups(vmm_temp4, vmm_temp3);
                    cmpps(vmm_temp4, vmm_iou_threshold, 0x07);  // order compare, 0 for unorders
                    // shoud not spoil iou(temp3), used in continued soft nms
                    uni_vmovups(vmm_temp2, vmm_temp3);
                    cmpps(vmm_temp2, vmm_iou_threshold, 0x05);   // _CMP_GE_US on sse, no direct _CMP_GE_OS supported.

                    uni_vandps(vmm_temp4, vmm_temp4, vmm_temp2);
                    uni_vtestps(vmm_temp4, vmm_temp4);  // ptest: bitwise check if all zeros, on sse41
                }
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

            // cmpps(_CMP_LE_OS) if new score is less or equal than score_threshold, suppressed.
            if (mayiuse(cpu::x64::avx512_common)) {
                vcmpps(k_mask, vmm_temp3, vmm_score_threshold, 0x02); // vcmpps w/ kmask only on V5, w/o kmask version N/A on V5
                kandw(k_mask, k_mask, k_mask_one);
                kortestw(k_mask, k_mask);    // bitwise check if all zero
            } else if (mayiuse(cpu::x64::avx)) {
                vcmpps(vmm_temp4, vmm_temp3, vmm_score_threshold, 0x02);
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);
            } else {
                uni_vmovups(vmm_temp4, vmm_temp3);
                cmpps(vmm_temp4, vmm_score_threshold, 0x07);  // order compare, 0 for unorders
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);

                jz(main_loop_label, T_NEAR);

                cmpps(vmm_temp3, vmm_score_threshold, 0x02);  // _CMP_LE_US on sse
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp3.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);
            }

            jz(main_loop_label, T_NEAR);

            uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

            jmp(terminate_label, T_NEAR);
        }
        L(main_loop_end_label);

        step = 1;
        L(tail_loop_label);
        {
            cmp(reg_boxes_num, 1);
            jl(terminate_label, T_NEAR);

            sub(reg_boxes_coord0, step * sizeof(float));
            sub(reg_boxes_coord1, step * sizeof(float));
            sub(reg_boxes_coord2, step * sizeof(float));
            sub(reg_boxes_coord3, step * sizeof(float));

            iou(step);
            sub(reg_boxes_num, step);

            // soft suppressed by iou_threshold
            if (jcp.is_soft_suppressed_by_iou) {
                if (mayiuse(cpu::x64::avx512_common)) {
                    vcmpps(k_mask, vmm_temp3, vmm_iou_threshold, 0x0D); // vcmpps w/ kmask only on V5
                    kandw(k_mask, k_mask, k_mask_one);
                    kortestw(k_mask, k_mask);    // bitwise check if all zero
                } else if (mayiuse(cpu::x64::avx)) {
                    // vex instructions with xmm on avx and ymm on avx2
                    vcmpps(vmm_temp4, vmm_temp3, vmm_iou_threshold, 0x0D);  // xmm and ymm only on V1.
                    uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                    test(reg_temp_32, reg_temp_32);
                } else {
                    uni_vmovups(vmm_temp4, vmm_temp3);

                    cmpps(vmm_temp4, vmm_iou_threshold, 0x07);  // order compare
                    uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                    test(reg_temp_32, reg_temp_32);

                    jz(tail_loop_label_soft, T_NEAR);

                    uni_vmovups(vmm_temp2, vmm_temp3);
                    cmpps(vmm_temp2, vmm_iou_threshold, 0x05);  // _CMP_GE_US on sse
                    uni_vpextrd(reg_temp_32, Xmm(vmm_temp2.getIdx()), 0);
                    test(reg_temp_32, reg_temp_32);
                }

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
            if (mayiuse(cpu::x64::avx512_common)) {
                vcmpps(k_mask, vmm_temp3, vmm_score_threshold, 0x02); // vcmpps w/ kmask only on V5, w/o kmask version N/A on V5
                kandw(k_mask, k_mask, k_mask_one);
                kortestw(k_mask, k_mask);    // bitwise check if all zero
            } else if (mayiuse(cpu::x64::avx)) {
                vcmpps(vmm_temp4, vmm_temp3, vmm_score_threshold, 0x02);
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);
            } else {
                uni_vmovups(vmm_temp4, vmm_temp3);
                cmpps(vmm_temp4, vmm_score_threshold, 0x07);  // order compare, 0 for unorders
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp4.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);

                jz(tail_loop_label, T_NEAR);

                cmpps(vmm_temp3, vmm_score_threshold, 0x02);
                uni_vpextrd(reg_temp_32, Xmm(vmm_temp3.getIdx()), 0);
                test(reg_temp_32, reg_temp_32);
            }

            jz(tail_loop_label, T_NEAR);

            uni_vpextrd(ptr[reg_candidate_status], Xmm(vmm_zero.getIdx()), 0);

            jmp(terminate_label, T_NEAR);
        }

        L(terminate_label);
    }

    inline void iou(int ele_num) {
        auto load = [&](Xbyak::Reg64 reg_src, Vmm vmm_dst) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())},
                std::make_shared<load_emitter_context>(Precision::FP32, Precision::FP32, ele_num),
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

    // should rebase to use oneDNN after related merged
    void uni_vmovshdup(const Xbyak::Xmm &x, const Xbyak::Xmm &y) {
        if (mayiuse(cpu::x64::avx))
            vmovshdup(x, y);
        else
            movshdup(x, y);
    }

    void uni_vmovhlps(const Xbyak::Xmm &x, const Xbyak::Xmm &y, const Xbyak::Xmm &z) {
        if (mayiuse(cpu::x64::avx)) {
            vmovhlps(x, y, z);
        } else {
            assert(x.getIdx() == y.getIdx());
            movhlps(x, z);
        }
    }
};

} // namespace MKLDNNPlugin
