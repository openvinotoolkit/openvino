// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <memory>
#include <vector>
#include <mkldnn_types.h>
#include <ie_parallel.hpp>
#include <mkldnn_extension_utils.h>
#include "cpu_memcpy.h"
#include "utils/bfloat16.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "emitters/jit_load_store_emitters.hpp"

using namespace InferenceEngine;
using namespace MKLDNNPlugin;
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

struct jit_nms_config_params {
    NMSBoxEncodeType box_encode_type;
};

struct jit_nms_args {
    const void* selected_boxes_coord[BOX_COORD_NUM];
    size_t selected_boxes_num;
    const void* candidate_box;
    const void* iou_threshold;
    void *is_valid;
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

        this->preamble();

        mov(reg_table, l_table_constant);
        if (mayiuse(cpu::x64::avx512_common)) {
            kmovw(k_mask_one, word[reg_table + vlen]);
        }

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        mov(reg_boxes_coord0, ptr[reg_params + GET_OFF(selected_boxes_coord[0])]);
        mov(reg_boxes_coord1, ptr[reg_params + GET_OFF(selected_boxes_coord[0]) + 1 * sizeof(size_t)]);
        mov(reg_boxes_coord2, ptr[reg_params + GET_OFF(selected_boxes_coord[0]) + 2 * sizeof(size_t)]);
        mov(reg_boxes_coord3, ptr[reg_params + GET_OFF(selected_boxes_coord[0]) + 3 * sizeof(size_t)]);
        mov(reg_candidate_box, ptr[reg_params + GET_OFF(candidate_box)]);
        mov(reg_valid, ptr[reg_params + GET_OFF(is_valid)]);
        mov(reg_boxes_num, ptr[reg_params + GET_OFF(selected_boxes_num)]);
        mov(reg_iou_threshold, ptr[reg_params + GET_OFF(iou_threshold)]);
        uni_vbroadcastss(vmm_iou_threshold, ptr[reg_iou_threshold]);

        // exec
        // check from last to first
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
            uni_vmulps(vmm_temp1, vmm_candidate_coord3, ptr[reg_table]);   // height/2
            uni_vsubps(vmm_temp2, vmm_candidate_coord1, vmm_temp1);  // y_center - height/2
            uni_vaddps(vmm_temp3, vmm_candidate_coord1, vmm_temp1);  // y_center + height/2

            uni_vmulps(vmm_temp1, vmm_candidate_coord2, ptr[reg_table]);   // width/2
            uni_vsubps(vmm_temp4, vmm_candidate_coord0, vmm_temp1);  // x_center - width/2
            uni_vaddps(vmm_temp5, vmm_candidate_coord0, vmm_temp1);  // x_center + width/2

            uni_vmovups(vmm_candidate_coord0, vmm_temp2);
            uni_vmovups(vmm_candidate_coord2, vmm_temp3);
            uni_vmovups(vmm_candidate_coord1, vmm_temp4);
            uni_vmovups(vmm_candidate_coord3, vmm_temp5);
        }

        imul(reg_offset, reg_boxes_num, sizeof(float));
        add(reg_boxes_coord0, reg_offset);  // y1
        add(reg_boxes_coord1, reg_offset);  // x1
        add(reg_boxes_coord2, reg_offset);  // y2
        add(reg_boxes_coord3, reg_offset);  // x2

        int step = vlen / sizeof(float);
        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label terminate_label;
        Xbyak::Label suppressed_label;
        L(main_loop_label);
        {
            cmp(reg_boxes_num, step);
            jl(main_loop_end_label, T_NEAR);

            sub(reg_boxes_coord0, step * sizeof(float));
            sub(reg_boxes_coord1, step * sizeof(float));
            sub(reg_boxes_coord2, step * sizeof(float));
            sub(reg_boxes_coord3, step * sizeof(float));

            // result is in vmm_temp3
            iou(step);

            sub(reg_boxes_num, step);

            if (mayiuse(cpu::x64::avx512_common)) {
                vcmpps(k_mask, vmm_temp3, vmm_iou_threshold, 0x0D); // _CMP_GE_OS. vcmpps w/ kmask only on V5
                kortestw(k_mask, k_mask);    // bitwise check if all zero
            } else if (mayiuse(cpu::x64::avx2)) {
                vcmpps(vmm_temp4, vmm_temp3, vmm_iou_threshold, 0x0D);  // xmm and ymm only on V1.
                uni_vtestps(vmm_temp4, vmm_temp4);  // vtestps: sign bit check if all zeros, ymm and xmm only on V1, N/A on V5
            } else {
                cmpps(vmm_temp3, vmm_iou_threshold, 0x0D);   // cmpps w/o mask on sse41
                uni_vtestps(vmm_temp3, vmm_temp3);  // ptest: bitwise check if all zeros, on sse41
            }
            // if zero continue, else set result to suppressed and terminate
            jz(main_loop_label, T_NEAR);

            uni_vpextrd(ptr[reg_valid], Xmm(vmm_zero.getIdx()), 0);

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

            // iou result is in vmm_temp3
            iou(step);

            sub(reg_boxes_num, step);

            if (mayiuse(cpu::x64::avx512_common)) {
                vcmpps(k_mask, vmm_temp3, vmm_iou_threshold, 0x0D); // vcmpps w/ kmask only on V5
                kandw(k_mask, k_mask, k_mask_one);
                kortestw(k_mask, k_mask);    // bitwise check if all zero
            } else if (mayiuse(cpu::x64::avx2)) {
                vcmpps(vmm_temp4, vmm_temp3, vmm_iou_threshold, 0x0D);  // xmm and ymm only on V1.
                uni_vpextrd(reg_scalar_mask, Xmm(vmm_temp4.getIdx()), 0);
                test(reg_scalar_mask, reg_scalar_mask);
            } else {
                cmpps(vmm_temp3, vmm_iou_threshold, 0x0D);   // cmpps w/o mask on sse41
                uni_vpextrd(reg_scalar_mask, Xmm(vmm_temp3.getIdx()), 0);
                test(reg_scalar_mask, reg_scalar_mask);
            }

            jz(tail_loop_label, T_NEAR);

            uni_vpextrd(ptr[reg_valid], Xmm(vmm_zero.getIdx()), 0);

            jmp(terminate_label, T_NEAR);
        }

        L(terminate_label);

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();

        prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_boxes_coord0 = r8;
    Xbyak::Reg64 reg_boxes_coord1 = r9;
    Xbyak::Reg64 reg_boxes_coord2 = r10;
    Xbyak::Reg64 reg_boxes_coord3 = r11;
    Xbyak::Reg64 reg_candidate_box = r12;
    Xbyak::Reg64 reg_valid = r13;
    Xbyak::Reg64 reg_boxes_num = r14;
    Xbyak::Reg64 reg_iou_threshold = r15;

    Xbyak::Reg64 reg_load_table = rax;
    Xbyak::Reg64 reg_load_store_mask = rbx;
    Xbyak::Label l_table_constant;
    Xbyak::Reg64 reg_table = rsi;
    // rdx as unix default abi at very begining, could be resued here
    Xbyak::Reg64 reg_offset = rdx;
    Xbyak::Reg32 reg_scalar_mask = edx;

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
    Vmm vmm_temp5 = Vmm(13);

    Vmm vmm_zero = Vmm(15);
    Vmm vmm_iou_threshold = Vmm(14);

    Xbyak::Opmask k_mask = Xbyak::Opmask(1);
    Xbyak::Opmask k_mask_one = Xbyak::Opmask(2);

    inline void iou(int ele_num) {
        load_emitter->emit_code({static_cast<size_t>(reg_boxes_coord0.getIdx())}, {static_cast<size_t>(vmm_boxes_coord0.getIdx())},
            std::make_shared<load_emitter_context>(Precision::FP32, Precision::FP32, ele_num),
            {}, {load_pool_gpr_idxs});
        load_emitter->emit_code({static_cast<size_t>(reg_boxes_coord1.getIdx())}, {static_cast<size_t>(vmm_boxes_coord1.getIdx())},
            std::make_shared<load_emitter_context>(Precision::FP32, Precision::FP32, ele_num),
            {}, {load_pool_gpr_idxs});
        load_emitter->emit_code({static_cast<size_t>(reg_boxes_coord2.getIdx())}, {static_cast<size_t>(vmm_boxes_coord2.getIdx())},
            std::make_shared<load_emitter_context>(Precision::FP32, Precision::FP32, ele_num),
            {}, {load_pool_gpr_idxs});
        load_emitter->emit_code({static_cast<size_t>(reg_boxes_coord3.getIdx())}, {static_cast<size_t>(vmm_boxes_coord3.getIdx())},
            std::make_shared<load_emitter_context>(Precision::FP32, Precision::FP32, ele_num),
            {}, {load_pool_gpr_idxs});

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
            uni_vmulps(vmm_temp1, vmm_boxes_coord3, ptr[reg_table]);   // height/2
            uni_vsubps(vmm_temp2, vmm_boxes_coord1, vmm_temp1);  // y_center - height/2
            uni_vaddps(vmm_temp3, vmm_boxes_coord1, vmm_temp1);  // y_center + height/2

            uni_vmulps(vmm_temp1, vmm_boxes_coord2, ptr[reg_table]);   // width/2
            uni_vsubps(vmm_temp4, vmm_boxes_coord0, vmm_temp1);  // x_center - width/2
            uni_vaddps(vmm_temp5, vmm_boxes_coord0, vmm_temp1);  // x_center + width/2

            uni_vmovups(vmm_boxes_coord0, vmm_temp2);
            uni_vmovups(vmm_boxes_coord2, vmm_temp3);
            uni_vmovups(vmm_boxes_coord1, vmm_temp4);
            uni_vmovups(vmm_boxes_coord3, vmm_temp5);
        }

        uni_vsubps(vmm_temp1, vmm_boxes_coord2, vmm_boxes_coord0);
        uni_vsubps(vmm_temp2, vmm_boxes_coord3, vmm_boxes_coord1);
        uni_vmulps(vmm_temp1, vmm_temp1, vmm_temp2);  // area boxes

        uni_vsubps(vmm_temp2, vmm_candidate_coord2, vmm_candidate_coord0);
        uni_vsubps(vmm_temp3, vmm_candidate_coord3, vmm_candidate_coord1);
        uni_vmulps(vmm_temp2, vmm_temp2, vmm_temp3);  // area candidate(bc)  // candidate area calculate once and check if 0

        // y of intersection
        uni_vminps(vmm_temp3, vmm_boxes_coord2, vmm_candidate_coord2);  // min(Ymax)
        uni_vmaxps(vmm_temp4, vmm_boxes_coord0, vmm_candidate_coord0);  // max(Ymin)
        uni_vsubps(vmm_temp3, vmm_temp3, vmm_temp4);  // min(Ymax) - max(Ymin)
        uni_vmaxps(vmm_temp3, vmm_temp3, vmm_zero);

        // x of intersection
        uni_vminps(vmm_temp4, vmm_boxes_coord3, vmm_candidate_coord3);  // min(Xmax)
        uni_vmaxps(vmm_temp5, vmm_boxes_coord1, vmm_candidate_coord1);  // max(Xmin)
        uni_vsubps(vmm_temp4, vmm_temp4, vmm_temp5);  // min(Xmax) - max(Xmin)
        uni_vmaxps(vmm_temp4, vmm_temp4, vmm_zero);

        // intersection_area
        uni_vmulps(vmm_temp3, vmm_temp3, vmm_temp4);

        // iou: intersection_area / (areaI + areaJ - intersection_area);
        uni_vaddps(vmm_temp1, vmm_temp1, vmm_temp2);
        uni_vsubps(vmm_temp1, vmm_temp1, vmm_temp3);
        uni_vdivps(vmm_temp3, vmm_temp3, vmm_temp1);
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

} // namespace MKLDNNPlugin
