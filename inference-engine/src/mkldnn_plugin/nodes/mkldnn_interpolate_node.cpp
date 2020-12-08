// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_interpolate_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_quantize_node.h"
#include <legacy/ie_layers.h>
#include "mkldnn_eltwise_node.h"
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <legacy/ie_layers_internal.hpp>
#include "ie_parallel.hpp"
#include <algorithm>

#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"
#include "jit_uni_quantization.hpp"
#include "common/cpu_memcpy.h"
#include "ngraph/type/bfloat16.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;
using namespace Xbyak;


#define GET_OFF(field) offsetof(jit_interpolate_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_interpolate_kernel_f32 : public jit_uni_interpolate_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_interpolate_kernel_f32)

    explicit jit_uni_interpolate_kernel_f32(jit_interpolate_config_params jcp, const mkldnn_primitive_attr &attr)
    : jit_uni_interpolate_kernel(jcp, attr), jit_generator() {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this,
                        post_op.eltwise.alg,
                        post_op.eltwise.alpha,
                        post_op.eltwise.beta));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this,
                        post_op.depthwise.alg));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        this->preamble();

        if (attr_.post_ops_.len_ != 0)
            mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);
        if (isa == cpu::avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        switch (jcp_.mode) {
            case InterpolateMode::nearest: {
                mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
                mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
                mov(reg_index, ptr[reg_params + GET_OFF(index)]);
                mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

                switch (jcp_.layout) {
                    case InterpolateLayoutType::planar: {
                        nn_planar();
                        break;
                    }
                    case InterpolateLayoutType::block: {
                        nn_blk();
                        break;
                    }
                    case InterpolateLayoutType::by_channel: {
                        nn_by_channel();
                        break;
                    }
                    default:
                        assert(!"unsupported memory layout for interpolate layer with nearest neighbor mode.");
                }
                break;
            }
            case InterpolateMode::linear_onnx: {
                switch (jcp_.layout) {
                    case InterpolateLayoutType::planar: {
                        linear_onnx_planar();
                        break;
                    }
                    case InterpolateLayoutType::block:
                    case InterpolateLayoutType::by_channel: {
                        linear_onnx_c_gathered();
                        break;
                    }
                    default:
                        assert(!"unsupported memory layout for interpolate layer with linear_onnx mode.");
                }
                break;
            }
            case InterpolateMode::cubic: {
                switch (jcp_.layout) {
                    case InterpolateLayoutType::planar: {
                        cubic_planar();
                        break;
                    }
                    case InterpolateLayoutType::block:
                    case InterpolateLayoutType::by_channel: {
                        cubic_c_gathered();
                        break;
                    }
                    default:
                        assert(!"unsupported memory layout for interpolate layer with cubic mode.");
                }
                break;
            }
            case InterpolateMode::linear: {
                assert(!"unsupported mode for interpolate layer with JITTED implimentation.");
                break;
            }
            default: {
                assert(!"unsupported mode for interpolate layer.");
            }
        }

        this->postamble();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
        if ((jcp_.mode == InterpolateMode::cubic) && (jcp_.layout == InterpolateLayoutType::planar)) {
            prepare_cubic_planar_table();
        }

        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_src_aux = r15;
    Xbyak::Reg64 reg_src_aux1 = r11;
    Xbyak::Reg64 reg_src_aux2 = r12;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_work_amount = r13;
    Xbyak::Reg64 reg_index = r14;
    Xbyak::Reg64 reg_params = abi_param1;

    Reg8 reg_tmp_8 = r10b;
    Reg32 reg_tmp_32 = r10d;
    Reg64 reg_tmp_64 = r10;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = rcx;
    Xbyak::Reg32 reg_index_offset = edx;

    // for cubic planar
    Xbyak::Reg64 reg_tbl_y = rsi;
    Xbyak::Reg64 reg_tbl_x = rbp;
    Xbyak::Reg64 reg_table = rdx;   // do not need reg_index_offset in this mode, so use rdx

    Vmm vmm_val = Vmm(0);
    Xmm xmm_val = Xmm(0);
    Vmm vmm_index = Vmm(1);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_mask = Vmm(3);
    Vmm vmm_d_weights = Vmm(4);
    Vmm vmm_d_bias = Vmm(5);

    // for linear
    Vmm vmm_weightT = Vmm(15);
    Xmm xmm_weightT = Xmm(15);
    Vmm vmm_weightB = Vmm(14);
    Xmm xmm_weightB = Xmm(14);
    Vmm vmm_weightL = Vmm(13);
    Xmm xmm_weightL = Xmm(13);
    Vmm vmm_weightR = Vmm(12);
    Xmm xmm_weightR = Xmm(12);
    Vmm vmm_valTL = Vmm(11);
    Xmm xmm_valTL = Xmm(11);
    Vmm vmm_valTR = vmm_val;
    Xmm xmm_valTR = xmm_val;
    Vmm vmm_valBL = Vmm(9);
    Xmm xmm_valBL = Xmm(9);
    Vmm vmm_valBR = Vmm(8);
    Xmm xmm_valBR = Xmm(8);

    // for cubic
    Vmm vmm_src = Vmm(6);
    Xmm xmm_src = Xmm(6);
    Vmm vmm_dstX = Vmm(7);

    Vmm vmm_weightX0 = vmm_weightT;
    Vmm vmm_weightX1 = vmm_weightB;
    Vmm vmm_weightX2 = vmm_weightL;
    Vmm vmm_weightX3 = vmm_weightR;
    Vmm vmm_weightY0 = vmm_valTL;
    Vmm vmm_weightY1 = Vmm(10);  // vmm_valTR is vmm_val, need reserved
    Vmm vmm_weightY2 = vmm_valBL;
    Vmm vmm_weightY3 = vmm_valBR;
    // cubic planar
    Vmm vmm_one = vmm_index;
    Vmm vmm_weightY = vmm_weightY0;
    Vmm vmm_index_y_itr = vmm_weightY1;
    Vmm vmm_index_x_itr = vmm_weightY2;
    Vmm vmm_tbl_y = vmm_weightY3;
    // temporally used. when post ops, value in vmm_d_weights and vmm_d_bias is re-loaded(init) each time.
    Vmm vmm_index_in_y = vmm_d_weights;
    Vmm vmm_index_in_x = vmm_d_bias;

    Xbyak::Label l_table_constant;
    Opmask k_mask = Xbyak::Opmask(1);

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    void nn_planar() {
        Xbyak::Reg64 reg_index_h = reg_src_aux1;
        Xbyak::Reg64 reg_index_w = reg_src_aux2;
        mov(reg_index_h, reg_index);
        // reg_index represent reg_index_w
        add(reg_index, jcp_.OH * jcp_.indices_size);
        // bk for reset to reg_index_w
        mov(reg_index_w, reg_index);

        Xbyak::Label out_loop_label;
        Xbyak::Label out_loop_end;

        Xbyak::Reg64 reg_work_amount_oh = rdi;
        mov(reg_work_amount_oh, jcp_.OH);
        L(out_loop_label);
        {
            // outloop status
            cmp(reg_work_amount_oh, 1);
            jl(out_loop_end, T_NEAR);

            //reset work_amount to OW
            mov(reg_work_amount, jcp_.OW);

            Xbyak::Reg64 reg_src_h = rsi;
            mov(reg_src_h, reg_src);
            // index_h * IW * dataSize done when built to avoid redundent compute
            mov(reg_index_offset, dword[reg_index_h]);
            add(reg_src_h, reg_index_offset);  // reg_src_h now point to begin of row

            // reset index_w, index_w * dataSize done when built to avoid redundent compute
            mov(reg_index, reg_index_w);
            int step = vlen / sizeof(float);

            Xbyak::Label nn_loop_label;
            Xbyak::Label nn_loop_end_label;
            Xbyak::Label nn_tail_loop_label;
            Xbyak::Label nn_tail_loop_end_label;

            L(nn_loop_label);   // inner loop
            {
                cmp(reg_work_amount, step);
                jl(nn_loop_end_label, T_NEAR);

                uni_vmovdqu(vmm_index, ptr[reg_index]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                vgatherdps(vmm_val, ptr[reg_src_h + vmm_index], vmm_mask);
                if (attr_.post_ops_.len_ != 0)
                    apply_post_ops(jcp_.dst_dt, 1);
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                add(reg_dst, step * jcp_.dst_data_size);
                add(reg_index, step * jcp_.indices_size);
                sub(reg_work_amount, step);

                jmp(nn_loop_label, T_NEAR);
            }
            L(nn_loop_end_label);

            step = 1;
            L(nn_tail_loop_label);
            {
                cmp(reg_work_amount, 1);
                jl(nn_tail_loop_end_label, T_NEAR);

                mov(reg_src_aux, reg_src_h);
                mov(reg_index_offset, dword[reg_index]);
                add(reg_src_aux, reg_index_offset);

                load_scalar(xmm_val, ptr[reg_src_aux], jcp_.src_dt);
                if (attr_.post_ops_.len_ != 0)
                    apply_post_ops(jcp_.dst_dt, 1);
                store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

                add(reg_dst, step * jcp_.dst_data_size);
                add(reg_index, step * jcp_.indices_size);
                sub(reg_work_amount, step);

                jmp(nn_tail_loop_label, T_NEAR);
            }
            L(nn_tail_loop_end_label);    // inner loop end

            //increment index_h to next row
            add(reg_index_h, jcp_.indices_size);

            sub(reg_work_amount_oh, 1);
            jmp(out_loop_label, T_NEAR);
        }
        L(out_loop_end);
    }

    void nn_blk() {
        int step = vlen / sizeof(float);
        if (isa == cpu::sse42)
            step *= 2;

        Xbyak::Label nn_loop_label;
        Xbyak::Label nn_loop_end_label;
        L(nn_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(nn_loop_end_label, T_NEAR);

            mov(reg_src_aux, reg_src);
            mov(reg_index_offset, dword[reg_index]);
            add(reg_src_aux, reg_index_offset);

            load_vector(vmm_val, ptr[reg_src_aux], jcp_.src_dt);
            if (attr_.post_ops_.len_ != 0)
                apply_post_ops(jcp_.dst_dt, 0);
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            if (isa == cpu::sse42) {
                int sse42_offset = 4;
                add(reg_src_aux, sse42_offset * jcp_.src_data_size);
                load_vector(vmm_val, ptr[reg_src_aux], jcp_.src_dt);
                if (attr_.post_ops_.len_ != 0) {
                    add(reg_oc_off, sse42_offset * sizeof(float));
                    apply_post_ops(jcp_.dst_dt, 0);
                    sub(reg_oc_off, sse42_offset * sizeof(float));
                }
                store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
            }

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_index, jcp_.indices_size);
            sub(reg_work_amount, 1);

            jmp(nn_loop_label, T_NEAR);
        }
        L(nn_loop_end_label);
    }

    void nn_by_channel() {
        // kernel for C * OW
        Xbyak::Label out_loop_label;
        Xbyak::Label out_loop_end;

        Xbyak::Reg64 reg_work_amount_out = reg_src_aux1;
        mov(reg_work_amount_out, jcp_.OW);
        L(out_loop_label);
        {
            cmp(reg_work_amount_out, 1);
            jl(out_loop_end, T_NEAR);

            int step = vlen / sizeof(float);

            //inner loop for C
            Xbyak::Label nn_loop_label;
            Xbyak::Label nn_loop_end_label;
            Xbyak::Label nn_tail_loop_label;
            Xbyak::Label nn_tail_loop_end_label;

            // inner loop for C
            // get current loop address reg_src_aux, from reg_src which is unchange, point this C * OW.
            // reset offset and work_amount.
            // dst and index address is continous, advanced each interator.
            mov(reg_src_aux, reg_src);
            // index*C*dataSize done when built to avoid redundent compute
            mov(reg_index_offset, dword[reg_index]);
            add(reg_src_aux, reg_index_offset);

            mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
            if (attr_.post_ops_.len_ != 0)
                mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

            L(nn_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(nn_loop_end_label, T_NEAR);

                load_vector(vmm_val, ptr[reg_src_aux], jcp_.src_dt);
                if (attr_.post_ops_.len_ != 0)
                    apply_post_ops(jcp_.dst_dt, 0);
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                add(reg_dst, step * jcp_.dst_data_size);
                add(reg_src_aux, step * jcp_.src_data_size);
                add(reg_oc_off, step * sizeof(float));
                sub(reg_work_amount, step);

                jmp(nn_loop_label, T_NEAR);
            }
            L(nn_loop_end_label);

            step = 1;
            L(nn_tail_loop_label);
            {
                cmp(reg_work_amount, 1);
                jl(nn_tail_loop_end_label, T_NEAR);

                load_scalar(xmm_val, ptr[reg_src_aux], jcp_.src_dt);
                if (attr_.post_ops_.len_ != 0)
                    apply_post_ops(jcp_.dst_dt, 0);
                store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

                add(reg_dst, step * jcp_.dst_data_size);
                add(reg_src_aux, step * jcp_.src_data_size);
                add(reg_oc_off, step * sizeof(float));
                sub(reg_work_amount, step);

                jmp(nn_tail_loop_label, T_NEAR);
            }
            L(nn_tail_loop_end_label);
            // inner loop end

            add(reg_index, jcp_.indices_size);

            sub(reg_work_amount_out, 1);
            jmp(out_loop_label, T_NEAR);
        }
        L(out_loop_end);
    }

    void linear_onnx_c_gathered() {
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);

        mov(reg_src, ptr[reg_params + GET_OFF(weight_ptr[0])]);
        mov(reg_src_aux, ptr[reg_params + GET_OFF(weight_ptr[0]) + sizeof(size_t)]);
        mov(reg_src_aux1, ptr[reg_params + GET_OFF(weight_ptr[0]) + 2 * sizeof(size_t)]);
        mov(reg_src_aux2, ptr[reg_params + GET_OFF(weight_ptr[0]) + 3 * sizeof(size_t)]);
        uni_vbroadcastss(vmm_weightL, ptr[reg_src]);
        uni_vbroadcastss(vmm_weightR, ptr[reg_src_aux]);
        uni_vbroadcastss(vmm_weightT, ptr[reg_src_aux1]);
        uni_vbroadcastss(vmm_weightB, ptr[reg_src_aux2]);

        mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
        mov(reg_src_aux, ptr[reg_params + GET_OFF(src_ptr[0]) + sizeof(size_t)]);
        mov(reg_src_aux1, ptr[reg_params + GET_OFF(src_ptr[0]) + 2 * sizeof(size_t)]);
        mov(reg_src_aux2, ptr[reg_params + GET_OFF(src_ptr[0]) + 3 * sizeof(size_t)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        int step = vlen / sizeof(float);
        int blk = (isa == cpu::sse42) ? (2 * step) : step;

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label blk_tail_loop_label;
        Xbyak::Label blk_tail_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;
        L(main_loop_label);
        {
            if (jcp_.layout == InterpolateLayoutType::by_channel) {
                cmp(reg_work_amount, step);
                jl(main_loop_end_label, T_NEAR);
            } else {
                cmp(reg_work_amount, 1);
                jl(tail_loop_end_label, T_NEAR);
            }

            load_vector(vmm_valTL, ptr[reg_src], jcp_.src_dt);
            load_vector(vmm_valTR, ptr[reg_src_aux], jcp_.src_dt);
            load_vector(vmm_valBL, ptr[reg_src_aux1], jcp_.src_dt);
            load_vector(vmm_valBR, ptr[reg_src_aux2], jcp_.src_dt);

            linear_onnx_worker();

            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, false);  // vmm_val is vmm_valTR
                add(reg_oc_off, step * sizeof(float));
            }
            store_vector(ptr[reg_dst], vmm_valTR, jcp_.dst_dt);

            if ((isa == cpu::sse42) && (jcp_.layout == InterpolateLayoutType::block)) {
                int sse42_offset = 4;  // vmm is xmm here
                load_vector(vmm_valTL, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                load_vector(vmm_valTR, ptr[reg_src_aux + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                load_vector(vmm_valBL, ptr[reg_src_aux1 + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                load_vector(vmm_valBR, ptr[reg_src_aux2 + sse42_offset * jcp_.src_data_size], jcp_.src_dt);

                linear_onnx_worker();

                if (attr_.post_ops_.len_ != 0) {
                    apply_post_ops(jcp_.dst_dt, false);
                    add(reg_oc_off, step * sizeof(float));
                }
                store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_valTR, jcp_.dst_dt);
            }
            if (jcp_.layout == InterpolateLayoutType::by_channel) {
                int dst_stride = step * jcp_.dst_data_size;
                int src_stride = step * jcp_.src_data_size;
                add(reg_dst, dst_stride);
                add(reg_src, src_stride);
                add(reg_src_aux, src_stride);
                add(reg_src_aux1, src_stride);
                add(reg_src_aux2, src_stride);
                sub(reg_work_amount, step);    // work_amount is c
            } else {
                int dst_stride = blk * jcp_.OW * jcp_.OH * jcp_.dst_data_size;
                int src_stride = blk * jcp_.IW * jcp_.IH * jcp_.src_data_size;
                add(reg_dst, dst_stride);
                add(reg_src, src_stride);
                add(reg_src_aux, src_stride);
                add(reg_src_aux1, src_stride);
                add(reg_src_aux2, src_stride);
                sub(reg_work_amount, 1);  // work_amount = div_up(c, blk), no tails
            }

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        step = 4;
        L(blk_tail_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(blk_tail_loop_end_label, T_NEAR);

            // use xmm for 4s in tails
            load_xmm(xmm_valTL, ptr[reg_src], jcp_.src_dt);
            load_xmm(xmm_valTR, ptr[reg_src_aux], jcp_.src_dt);
            load_xmm(xmm_valBL, ptr[reg_src_aux1], jcp_.src_dt);
            load_xmm(xmm_valBR, ptr[reg_src_aux2], jcp_.src_dt);

            linear_onnx_worker();

            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, false);  // vmm_val is vmm_valTR
                add(reg_oc_off, step * sizeof(float));
            }
            store_xmm(ptr[reg_dst], xmm_valTR, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_src, step * jcp_.src_data_size);
            add(reg_src_aux, step * jcp_.src_data_size);
            add(reg_src_aux1, step * jcp_.src_data_size);
            add(reg_src_aux2, step * jcp_.src_data_size);
            sub(reg_work_amount, step);

            jmp(blk_tail_loop_label, T_NEAR);
        }
        L(blk_tail_loop_end_label);

        step = 1;
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            // still use xmm on avx2/avx512
            load_scalar(xmm_valTL, ptr[reg_src], jcp_.src_dt);
            load_scalar(xmm_valTR, ptr[reg_src_aux], jcp_.src_dt);
            load_scalar(xmm_valBL, ptr[reg_src_aux1], jcp_.src_dt);
            load_scalar(xmm_valBR, ptr[reg_src_aux2], jcp_.src_dt);

            linear_onnx_worker();

            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, false);  // vmm_val is vmm_valTR
                add(reg_oc_off, step * sizeof(float));
            }
            store_scalar(ptr[reg_dst], xmm_valTR, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_src, step * jcp_.src_data_size);
            add(reg_src_aux, step * jcp_.src_data_size);
            add(reg_src_aux1, step * jcp_.src_data_size);
            add(reg_src_aux2, step * jcp_.src_data_size);
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    void linear_onnx_planar() {
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
        mov(reg_src_aux, ptr[reg_params + GET_OFF(weight_ptr[0])]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        int step = vlen / sizeof(float);
        int index_stride = jcp_.OW * jcp_.OH * jcp_.indices_size;
        int weight_stride = jcp_.OW * jcp_.OH * sizeof(float);

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;
        L(main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(main_loop_end_label, T_NEAR);

            uni_vmovdqu(vmm_index, ptr[reg_index]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_valTL, ptr[reg_src + vmm_index], vmm_mask);

            uni_vmovdqu(vmm_index, ptr[reg_index + index_stride]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_valTR, ptr[reg_src + vmm_index], vmm_mask);

            uni_vmovdqu(vmm_index, ptr[reg_index + 2 * index_stride]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_valBL, ptr[reg_src + vmm_index], vmm_mask);

            uni_vmovdqu(vmm_index, ptr[reg_index + 3 * index_stride]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_valBR, ptr[reg_src + vmm_index], vmm_mask);

            // reg_src_aux point to weight
            load_vector(vmm_weightL, ptr[reg_src_aux], memory::f32);
            load_vector(vmm_weightR, ptr[reg_src_aux + weight_stride], memory::f32);
            load_vector(vmm_weightT, ptr[reg_src_aux + 2 * weight_stride], memory::f32);
            load_vector(vmm_weightB, ptr[reg_src_aux + 3 * weight_stride], memory::f32);

            linear_onnx_worker();

            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, true);  // vmm_val is vmm_valTR, broadcase is true
            }
            store_vector(ptr[reg_dst], vmm_valTR, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_src_aux, step * sizeof(float));
            add(reg_index, step * jcp_.indices_size);
            sub(reg_work_amount, step);

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        step = 1;
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            // still use xmm on avx2/avx512
            mov(reg_src_aux1, reg_src);
            mov(reg_index_offset, dword[reg_index]);
            add(reg_src_aux1, reg_index_offset);
            load_scalar(xmm_valTL, ptr[reg_src_aux1], jcp_.src_dt);

            mov(reg_src_aux1, reg_src);
            mov(reg_index_offset, dword[reg_index + index_stride]);
            add(reg_src_aux1, reg_index_offset);
            load_scalar(xmm_valTR, ptr[reg_src_aux1], jcp_.src_dt);

            mov(reg_src_aux1, reg_src);
            mov(reg_index_offset, dword[reg_index + 2 * index_stride]);
            add(reg_src_aux1, reg_index_offset);
            load_scalar(xmm_valBL, ptr[reg_src_aux1], jcp_.src_dt);

            mov(reg_src_aux1, reg_src);
            mov(reg_index_offset, dword[reg_index + 3 * index_stride]);
            add(reg_src_aux1, reg_index_offset);
            load_scalar(xmm_valBR, ptr[reg_src_aux1], jcp_.src_dt);

            load_scalar(xmm_weightL, ptr[reg_src_aux], memory::f32);
            load_scalar(xmm_weightR, ptr[reg_src_aux + weight_stride], memory::f32);
            load_scalar(xmm_weightT, ptr[reg_src_aux + 2 * weight_stride], memory::f32);
            load_scalar(xmm_weightB, ptr[reg_src_aux + 3 * weight_stride], memory::f32);

            linear_onnx_worker();

            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, true);  // process on vmm_val, vmm_val is vmm_valTR, and bc
            }
            store_scalar(ptr[reg_dst], xmm_valTR, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_src_aux, step * sizeof(float));
            add(reg_index, step * jcp_.indices_size);
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    // weightT * (srcTL * weightL + srcTR * weightR) +
    // weightB * (srcBL * weightL + srcBR * weightR)
    inline void linear_onnx_worker() {
        uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightR);
        uni_vmulps(vmm_valBR, vmm_valBR, vmm_weightR);
        uni_vfmadd231ps(vmm_valTR, vmm_valTL, vmm_weightL);
        uni_vfmadd231ps(vmm_valBR, vmm_valBL, vmm_weightL);
        uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightT);
        uni_vfmadd231ps(vmm_valTR, vmm_valBR, vmm_weightB);
    }

    void cubic_c_gathered() {
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        // weight_ptr[0] point to weightX
        mov(reg_src_aux1, ptr[reg_params + GET_OFF(weight_ptr[0])]);
        uni_vbroadcastss(vmm_weightX0, ptr[reg_src_aux1]);
        uni_vbroadcastss(vmm_weightX1, ptr[reg_src_aux1 + 1 * sizeof(float)]);
        uni_vbroadcastss(vmm_weightX2, ptr[reg_src_aux1 + 2 * sizeof(float)]);
        uni_vbroadcastss(vmm_weightX3, ptr[reg_src_aux1 + 3 * sizeof(float)]);

        // weight_ptr[1] point to weightY
        mov(reg_src_aux1, ptr[reg_params + GET_OFF(weight_ptr[0]) + sizeof(size_t)]);
        uni_vbroadcastss(vmm_weightY0, ptr[reg_src_aux1]);
        uni_vbroadcastss(vmm_weightY1, ptr[reg_src_aux1 + 1 * sizeof(float)]);
        uni_vbroadcastss(vmm_weightY2, ptr[reg_src_aux1 + 2 * sizeof(float)]);
        uni_vbroadcastss(vmm_weightY3, ptr[reg_src_aux1 + 3 * sizeof(float)]);

        int step = vlen / sizeof(float);
        int blk = (isa == cpu::sse42) ? (2 * step) : step;

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;
        L(main_loop_label);
        {
            if (jcp_.layout == InterpolateLayoutType::by_channel) {
                cmp(reg_work_amount, step);
                jl(main_loop_end_label, T_NEAR);
            } else {
                cmp(reg_work_amount, 1);
                jl(tail_loop_end_label, T_NEAR);
            }

            uni_vpxor(vmm_val, vmm_val, vmm_val);

            cubic_c_gathered_matrix(false);

            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, false);     // vmm_val is default dst value to post_ops and store
                add(reg_oc_off, step * sizeof(float));
            }
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            if ((isa == cpu::sse42) && (jcp_.layout == InterpolateLayoutType::block)) {
                int sse42_offset = 4;  // vmm is xmm here
                add(reg_src, sse42_offset * jcp_.src_data_size);
                add(reg_dst, sse42_offset * jcp_.dst_data_size);

                uni_vpxor(vmm_val, vmm_val, vmm_val);

                cubic_c_gathered_matrix(false);

                if (attr_.post_ops_.len_ != 0) {
                    apply_post_ops(jcp_.dst_dt, false);
                    add(reg_oc_off, step * sizeof(float));  // second step for one blk
                }
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                sub(reg_src, sse42_offset * jcp_.src_data_size);
                sub(reg_dst, sse42_offset * jcp_.dst_data_size);
            }
            if (jcp_.layout == InterpolateLayoutType::by_channel) {
                int dst_stride = step * jcp_.dst_data_size;
                int src_stride = step * jcp_.src_data_size;
                add(reg_dst, dst_stride);
                add(reg_src, src_stride);
                sub(reg_work_amount, step);    // work_amount is c
            } else {
                int dst_stride = blk * jcp_.OW * jcp_.OH * jcp_.dst_data_size;
                int src_stride = blk * jcp_.IW * jcp_.IH * jcp_.src_data_size;
                add(reg_dst, dst_stride);
                add(reg_src, src_stride);
                sub(reg_work_amount, 1);  // work_amount = div_up(c, blk), no tails
            }

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        // only for by_channel layout for tails.
        step = 1;
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            // store final computed value
            uni_vpxor(vmm_val, vmm_val, vmm_val);

            cubic_c_gathered_matrix(true);

            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, false);     // vmm_val is default dst value
                add(reg_oc_off, step * sizeof(float));
            }
            store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

            int dst_stride = step * jcp_.dst_data_size;
            int src_stride = step * jcp_.src_data_size;
            add(reg_dst, dst_stride);
            add(reg_src, src_stride);
            sub(reg_work_amount, step);    // work_amount is c

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    inline void cubic_c_gathered_matrix(bool is_scalar) {
        // y0:  (x0 * weightX0 + x1 * weightX1 + x2 * weightX2 + x3 * weightX3) * weightY0
        cubic_c_gathered_line(0, vmm_weightY0, is_scalar);
        // y1
        cubic_c_gathered_line(4, vmm_weightY1, is_scalar);
        // y2
        cubic_c_gathered_line(8, vmm_weightY2, is_scalar);
        // y3
        cubic_c_gathered_line(12, vmm_weightY3, is_scalar);
    }

    inline void cubic_c_gathered_line(int index_start, Vmm vmm_weight, bool is_scalar) {
        uni_vpxor(vmm_dstX, vmm_dstX, vmm_dstX);
        cubic_c_gathered_pixel(index_start, vmm_weightX0, is_scalar);
        cubic_c_gathered_pixel(index_start + 1, vmm_weightX1, is_scalar);
        cubic_c_gathered_pixel(index_start + 2, vmm_weightX2, is_scalar);
        cubic_c_gathered_pixel(index_start + 3, vmm_weightX3, is_scalar);
        uni_vfmadd231ps(vmm_val, vmm_dstX, vmm_weight);
    }

    inline void cubic_c_gathered_pixel(int i, Vmm vmm_weight, bool is_scalar) {
        mov(reg_src_aux, reg_src);
        mov(reg_index_offset, dword[reg_index + i * jcp_.indices_size]);
        add(reg_src_aux, reg_index_offset);
        if (!is_scalar) {
            load_vector(vmm_src, ptr[reg_src_aux], jcp_.src_dt);
        } else {
            load_scalar(xmm_src, ptr[reg_src_aux], jcp_.src_dt);
        }
        uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weight);
    }

    void cubic_planar() {
        mov(reg_table, l_table_constant);
        // src_ptr[2] for oh sequence, src_ptr[3] for ow sequence
        mov(reg_tbl_y, ptr[reg_params + GET_OFF(src_ptr[0]) + 2 * sizeof(size_t)]);
        mov(reg_tbl_x, ptr[reg_params + GET_OFF(src_ptr[0]) + 3 * sizeof(size_t)]);
        uni_vmovdqu(vmm_one, cubic_planar_table_val(0));
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
        // index_OW
        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
        // index_OH from src_ptr[1]
        Xbyak::Reg64 reg_index_y = reg_src_aux;
        mov(reg_index_y, ptr[reg_params + GET_OFF(src_ptr[0]) + sizeof(size_t)]);
        // weight_OW
        Xbyak::Reg64 reg_weight_x = reg_src_aux1;
        mov(reg_weight_x, ptr[reg_params + GET_OFF(weight_ptr[0])]);
        // weight_OH
        Xbyak::Reg64 reg_weight_y = reg_src_aux2;
        mov(reg_weight_y, ptr[reg_params + GET_OFF(weight_ptr[0]) + sizeof(size_t)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        int step = vlen / sizeof(float);
        int grid_len = 4;

        // 0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19
        // 20  21  22  23  24  25  26  27  28  29  30   31   32   33   34   35   36   37   38   39
        // for 3th step(8): 16  17  18  19  20  21  22  23
        //               y: 0   0   0   0   1   1   1   1
        //               x: 16  17  18  19  0   1   2   3

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;
        L(main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(main_loop_end_label, T_NEAR);

            // vmm_tbl_y: (0 0 0 0 1 1 1 1 * index_size) --> (0 0 0 0 4 4 4 4)
            uni_vmovdqu(vmm_tbl_y, ptr[reg_tbl_y]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            // vmm_index_in_y: 0 0 0 0 2 2 2 2
            vpgatherdd(vmm_index_in_y, ptr[reg_index_y + vmm_tbl_y], vmm_mask);

            // use vmm_val temporally for value in reg_tbl_x: 16  17  18  19  0   1   2   3
            uni_vmovdqu(vmm_val, ptr[reg_tbl_x]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            // e.g. vmm_index_in_x: 32 34 36 38 0 2 4 6, now save src index.
            vpgatherdd(vmm_index_in_x, ptr[reg_index + vmm_val], vmm_mask);

            // build weightX used in y0-y3
            // weight format: w0_0 w1_0 w2_0 w3_0 w0_1 w1_1 w2_1 w3_1 ...
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightX0, ptr[reg_weight_x + vmm_val * grid_len], vmm_mask);  // 4 in vmm_val for weight_size, another 4 for grid_len

            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            // shift weight_size then gather second weight
            vgatherdps(vmm_weightX1, ptr[reg_weight_x + sizeof(float) + (vmm_val * grid_len)], vmm_mask);

            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightX2, ptr[reg_weight_x + 2 * sizeof(float) + (vmm_val * grid_len)], vmm_mask);

            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightX3, ptr[reg_weight_x + 3 * sizeof(float) + (vmm_val * grid_len)], vmm_mask);
            // vmm_val is now relieved and used for dst_value

            uni_vpxor(vmm_val, vmm_val, vmm_val);
            // y0
            vpsubd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);

            // weight y0
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightY, ptr[reg_weight_y + (vmm_tbl_y * grid_len)], vmm_mask);
            cubic_planar_line(false);

            // y1
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_in_y, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y1: shift weight_size
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightY, ptr[reg_weight_y + sizeof(float) + (vmm_tbl_y * grid_len)], vmm_mask);
            cubic_planar_line(false);

            // y2
            vpaddd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y2
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightY, ptr[reg_weight_y + 2 * sizeof(float) + (vmm_tbl_y * grid_len)], vmm_mask);
            cubic_planar_line(false);

            // y3
            vpaddd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            vpaddd(vmm_index_y_itr, vmm_index_y_itr, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y3
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightY, ptr[reg_weight_y + 3 * sizeof(float) + (vmm_tbl_y * grid_len)], vmm_mask);
            cubic_planar_line(false);

            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, true);  // oc_off is broadcast and always the same value for this channel
            }
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            add(reg_tbl_y, step * sizeof(int));  // sizeof(int): sequence by dd()
            add(reg_tbl_x, step * sizeof(int));
            add(reg_dst, step * jcp_.dst_data_size);

            sub(reg_work_amount, step);

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        step = 1;
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            // get idx for input
            movss(Xmm(vmm_tbl_y.getIdx()), ptr[reg_tbl_y]);
            gather_i32_indices(vmm_index_in_y, reg_index_y, 0, vmm_tbl_y, 1, memory::s32, true);

            movss(Xmm(vmm_val.getIdx()), ptr[reg_tbl_x]);
            gather_i32_indices(vmm_index_in_x, reg_index, 0, vmm_val, 1, memory::s32, true);
            // gather weightX by input idx, used in y0-y3
            gather_i32_indices(vmm_weightX0, reg_weight_x, 0, vmm_val, grid_len, memory::f32, true);
            gather_i32_indices(vmm_weightX1, reg_weight_x, sizeof(float), vmm_val, grid_len, memory::f32, true);
            gather_i32_indices(vmm_weightX2, reg_weight_x, 2 * sizeof(float), vmm_val, grid_len, memory::f32, true);
            gather_i32_indices(vmm_weightX3, reg_weight_x, 3 * sizeof(float), vmm_val, grid_len, memory::f32, true);
            // vmm_val is now relieved and used for dst_value

            uni_vpxor(vmm_val, vmm_val, vmm_val);
            // y0
            vpsubd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);

            gather_i32_indices(vmm_weightY, reg_weight_y, 0, vmm_tbl_y, grid_len, memory::f32, true);
            cubic_planar_line(true);

            // y1
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_in_y, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y1: shift weight_size
            gather_i32_indices(vmm_weightY, reg_weight_y, sizeof(float), vmm_tbl_y, grid_len, memory::f32, true);
            cubic_planar_line(true);

            // y2
            vpaddd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y2
            gather_i32_indices(vmm_weightY, reg_weight_y, 2 * sizeof(float), vmm_tbl_y, grid_len, memory::f32, true);
            cubic_planar_line(true);

            // y3
            vpaddd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            vpaddd(vmm_index_y_itr, vmm_index_y_itr, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y3
            gather_i32_indices(vmm_weightY, reg_weight_y, 3 * sizeof(float), vmm_tbl_y, grid_len, memory::f32, true);
            cubic_planar_line(true);

            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, true);  // oc_off is broadcast and always the same value for this channel
            }
            store_scalar(ptr[reg_dst], Xmm(vmm_val.getIdx()), jcp_.dst_dt);

            add(reg_tbl_y, step * sizeof(int));  // sizeof(int): sequence with dd()
            add(reg_tbl_x, step * sizeof(int));
            add(reg_dst, step * jcp_.dst_data_size);

            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    inline void cubic_planar_line(bool is_scalar) {
        uni_vpxor(vmm_dstX, vmm_dstX, vmm_dstX);
        cubic_planar_pixel(0, is_scalar);
        cubic_planar_pixel(1, is_scalar);
        cubic_planar_pixel(2, is_scalar);
        cubic_planar_pixel(3, is_scalar);
        uni_vfmadd231ps(vmm_val, vmm_dstX, vmm_weightY);
    }

    inline void cubic_planar_pixel(int itr, bool is_scalar) {
        // vmm_index_in_x have index for src
        if (itr == 0) {
            vpsubd(vmm_index_x_itr, vmm_index_in_x, vmm_one);
        } else if (itr == 1) {
            vpaddd(vmm_index_x_itr, vmm_index_in_x, vmm_zero);
        } else if (itr == 2) {
            vpaddd(vmm_index_x_itr, vmm_index_in_x, vmm_one);
        } else if (itr == 3) {
            vpaddd(vmm_index_x_itr, vmm_index_in_x, vmm_one);
            vpaddd(vmm_index_x_itr, vmm_index_x_itr, vmm_one);
        }

        // crop to [0, IW - 1]
        vpminsd(vmm_index_x_itr, vmm_index_x_itr, cubic_planar_table_val(2));
        vpmaxsd(vmm_index_x_itr, vmm_index_x_itr, vmm_zero);

        // value
        // index is: ptr[reg_src + (vmm_index_y_itr * jcp_.IW + vmm_index_x_itr) * jcp_.src_data_size]
        uni_vmovdqu(vmm_mask, cubic_planar_table_val(2));
        vpaddd(vmm_mask, vmm_mask, vmm_one);  // (IW - 1) + 1 = IW
        uni_vpmulld(vmm_mask, vmm_mask, vmm_index_y_itr);
        uni_vpaddd(vmm_index_x_itr, vmm_index_x_itr, vmm_mask);
        gather_i32_indices(vmm_src, reg_src, 0, vmm_index_x_itr, jcp_.src_data_size, memory::f32, is_scalar);

        if (itr == 0) {
            uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weightX0);
        } else if (itr == 1) {
            uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weightX1);
        } else if (itr == 2) {
            uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weightX2);
        } else if (itr == 3) {
            uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weightX3);
        }
    }

    inline void prepare_cubic_planar_table() {
        auto broadcast_int = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(int); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table_constant);
        broadcast_int(vals_for_cubic_planar.int_one);
        broadcast_int(jcp_.IH - 1);
        broadcast_int(jcp_.IW - 1);
        dd(vals_for_cubic_planar.mask_gather_avx512);
    }

    struct vals_for_cubic_planar_type {
        int int_one = 0x00000001;
        int mask_gather_avx512 = 0x0000ffff;  // 00000000000000001111111111111111
    } vals_for_cubic_planar;

    inline Xbyak::Address cubic_planar_table_val(int index) {
        return ptr[reg_table + index * vlen];
    }

    // always gather to Vmm, compute with Vmm, store with Xmm if scalar
    inline void gather_i32_indices(Vmm vmm_src, const Xbyak::Reg64 &base, int offset, Vmm vmm_indices, int scale,
                                memory::data_type src_dt, bool is_scalar) {
        Xbyak::Address table_idx = ptr[base + offset + vmm_indices * scale];
        if ((isa == cpu::avx512_common) && !is_scalar) {
            // [0-15] bit of int to mask
            kmovw(k_mask, cubic_planar_table_val(3));
            if (src_dt == memory::f32) {
                vgatherdps(vmm_src | k_mask, table_idx);  // dword index, packed single data
            } else if (src_dt == memory::s32) {
                vpgatherdd(vmm_src | k_mask, table_idx);  // dword index, dword data
            }
        } else if ((isa == cpu::avx2) && !is_scalar) {
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            if (src_dt == memory::f32) {
                vgatherdps(vmm_src, table_idx, vmm_mask);
            } else if (src_dt == memory::s32) {
                vpgatherdd(vmm_src, table_idx, vmm_mask);
            }
        } else {
            const int gpr_size = 8;
            sub(rsp, gpr_size);
            // move content in register to content in address(ptr[])
            mov(ptr[rsp], reg_tmp_64);

            // replace index with value in stack
            sub(rsp, vlen);
            uni_vmovdqu(ptr[rsp], vmm_indices);

            int repeats = is_scalar ? 1 : vlen / sizeof(float);
            for (size_t i = 0; i < repeats; ++i) {
                mov(reg_tmp_64.cvt32(), ptr[rsp + i * sizeof(int)]);       // sizeof(int)  index_size
                table_idx = ptr[base + offset + reg_tmp_64 * scale];       // scale: sizeof(float)   value_size
                mov(reg_tmp_64.cvt32(), table_idx);
                mov(ptr[rsp + i * sizeof(int)], reg_tmp_64.cvt32());
            }

            uni_vmovups(vmm_src, ptr[rsp]);
            add(rsp, vlen);
            // restore GPR state
            mov(reg_tmp_64, ptr[rsp]);
            add(rsp, gpr_size);
        }
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(vmm_src, op);
                break;
            case memory::s8:
                uni_vpmovsxbd(vmm_src, op);
                break;
            case memory::u8:
                uni_vpmovzxbd(vmm_src, op);
                break;
            case memory::bf16:
                uni_vpmovzxwd(vmm_src, op);
                uni_vpslld(vmm_src, vmm_src, 16);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != memory::f32 && src_dt != data_type::bf16)
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }

    inline void load_xmm(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(xmm_src, op);
                break;
            case memory::s8:
                uni_vpmovsxbd(xmm_src, op);
                break;
            case memory::u8:
                uni_vpmovzxbd(xmm_src, op);
                break;
            case memory::bf16:
                uni_vpmovzxwd(xmm_src, op);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != memory::f32 && src_dt != data_type::bf16)
            uni_vcvtdq2ps(xmm_src, xmm_src);
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                movss(xmm_src, op);
                break;
            case memory::s8:
                movsx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case memory::u8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case memory::bf16:
                pinsrw(xmm_src, op, 0x0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != data_type::f32 && src_dt != data_type::bf16) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());

        if (dst_dt == memory::f32) {
            uni_vmovups(op, vmm_dst);
        } else if (dst_dt == memory::u8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::avx512_common) {
                vpmaxsd(vmm_dst, vmm_dst, vmm_zero);
                vpmovusdb(op, vmm_dst);
            } else {
                uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::sse42)
                    vpermq(ymm_dst, ymm_dst, 0x08);
                uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
        } else if (dst_dt == memory::s8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::avx512_common) {
                vpmovsdb(op, vmm_dst);
            } else {
                uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::sse42)
                    vpermq(ymm_dst, ymm_dst, 0x08);
                uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
        } else if (dst_dt == memory::bf16) {
            if (mayiuse(avx512_core_bf16)) {
                vcvtneps2bf16(ymm_dst, vmm_dst);
                uni_vmovups(op, ymm_dst);
            } else {
                assert(!"data type of bf16 is only supported for ISA:avx512_core_bf16");
            }
        }
    }

    inline void store_xmm(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt) {
        if (dst_dt != memory::f32 && dst_dt != memory::bf16) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(op, xmm_dst);
                break;
            case memory::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movd(op, xmm_dst);
                break;
            case memory::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movd(op, xmm_dst);
                break;
            case memory::bf16:
                pshuflw(xmm_dst, xmm_dst, 0x0d);  // 01 01 01 01 --> 01 01 11 00  imm=0b00001101
                pshufhw(xmm_dst, xmm_dst, 0x0d);  // 01 01 11 00 --> 11 00 11 00
                pshufd(xmm_dst, xmm_dst, 0x08);   // 11 00 11 00 --> 11 11 00 00  imm=0b00001000
                vmovq(op, xmm_dst);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt) {
        if (dst_dt != data_type::f32 && dst_dt != data_type::bf16) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                movss(op, xmm_dst);
                break;
            case memory::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::bf16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                pextrw(op, xmm_dst, 0x0);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    // is_broadcast for broadcasting param for depth_wise and quantize(channel-sensitive post-ops), for fusion with plain layout.
    void apply_post_ops(memory::data_type dst_dt, bool is_broadcast) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        for (int i = 0; i < p.len_; i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));
                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);
                // weight and bias is padded. scalar as vector.
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1, reg_d_weights, reg_d_bias, is_broadcast);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_dt == memory::f32 || i != p.len_ - 1;

                int s_idx = vmm_val.getIdx();

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, 0, is_broadcast);

                if (do_dequantization) {
                    quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0, 0, is_broadcast);
                }

                quantization_inj_idx++;
            }
        }
    }
};

MKLDNNInterpolateNode::MKLDNNInterpolateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

// shapeND: n     c     d     h    w
// blockND: ncdhw cdhw  dhw   hw   w    1
// index  : 0      1    2     3    4    5
SizeVector getBlockND(SizeVector& shape) {
    int shapeRank = shape.size();
    SizeVector blockND(shapeRank + 1, 1);
    for (int i = shapeRank - 1; i >= 0; i--) {
        blockND[i] = shape[i] * blockND[i+1];
    }
    return blockND;
}

SizeVector to5Dim(SizeVector casesDim) {
    size_t caseSize = casesDim.size();
    SizeVector dim5(5, 1lu);
    if (caseSize > 2) {
        dim5[0] = casesDim[0];
        dim5[1] = casesDim[1];
    }
    if (caseSize == 5) {
        dim5[2] = casesDim[2];
    }
    dim5[3] = casesDim[caseSize - 2];
    dim5[4] = casesDim[caseSize - 1];
    return dim5;
}

void MKLDNNInterpolateNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 3 && getParentEdges().size() != 4)
        // data, target_shape, scale, axis(optional).
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' has incorrect number of input edges";
    isAxesSpecified = (getParentEdges().size() == 3) ? false : true;
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' has incorrect number of output edges";

    auto *layer = getCnnLayer().get();
    std::string modeString = layer->GetParamAsString("mode");
    if (modeString == "nearest") {
        mode = InterpolateMode::nearest;
    } else if (modeString == "linear") {
        mode = InterpolateMode::linear;
    } else if (modeString == "linear_onnx") {
        mode = InterpolateMode::linear_onnx;
    } else if (modeString == "cubic") {
        mode = InterpolateMode::cubic;
    } else {
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' does not support interpolate mode:" << modeString;
    }
    srcDim = getParentEdgeAt(DATA_ID)->getDims().ToSizeVector();
    int dataRank = srcDim.size();
    if (dataRank != 2 && dataRank != 4 && dataRank != 5) {
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() <<
        "' does not support input tensor of rank :" << dataRank;
    }
    if ((mode == InterpolateMode::cubic || mode == InterpolateMode::linear_onnx) && (dataRank == 5)) {
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() <<
        "' of 'linear_onnx' or 'cubic' mode only support input tensor of 2 or 4 rank";
    }

    modeString = layer->GetParamAsString("coordinate_transformation_mode", "half_pixel");
    if (modeString == "half_pixel") {
        coordTransMode = InterpolateCoordTransMode::half_pixel;
    } else if (modeString == "pytorch_half_pixel") {
        coordTransMode = InterpolateCoordTransMode::pytorch_half_pixel;
    } else if (modeString == "asymmetric") {
        coordTransMode = InterpolateCoordTransMode::asymmetric;
    } else if (modeString == "tf_half_pixel_for_nn") {
        coordTransMode = InterpolateCoordTransMode::tf_half_pixel_for_nn;
    } else if (modeString == "align_corners") {
        coordTransMode = InterpolateCoordTransMode::align_corners;
    } else {
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' does not support coordinate transformation mode: " << modeString;
    }

    if (mode == InterpolateMode::nearest) {
        modeString = layer->GetParamAsString("nearest_mode", "round_prefer_floor");
        if (modeString == "round_prefer_floor") {
            nearestMode = InterpolateNearestMode::round_prefer_floor;
        } else if (modeString == "round_prefer_ceil") {
            nearestMode = InterpolateNearestMode::round_prefer_ceil;
        } else if (modeString == "floor") {
            nearestMode = InterpolateNearestMode::floor;
        } else if (modeString == "ceil") {
            nearestMode = InterpolateNearestMode::ceil;
        } else if (modeString == "simple") {
            nearestMode = InterpolateNearestMode::simple;
        } else {
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' does not support nearest round mode: " << modeString;
        }
    } else if (mode == InterpolateMode::cubic) {
        cubeCoeff = layer->GetParamAsFloat("cube_coeff", -0.75);
    }
    antialias = layer->GetParamAsBool("antialias", false);
    shapeInferMode = layer->GetParamAsString("shape_calculation_mode");

    // get pad
    std::vector<int> defPad(dataRank, 0);
    padBegin = layer->GetParamAsInts("pads_begin", defPad);
    padEnd = layer->GetParamAsInts("pads_end", defPad);
    for (int i = 0; i < padBegin.size(); i++) {
        if (padBegin[i] != 0) {
            hasPad = true;
            break;
        }
    }
    for (int i = 0; i < padEnd.size(); i++) {
        if (padEnd[i] != 0) {
            hasPad = true;
            break;
        }
    }
    //correct pad
    if (hasPad) {
        auto correctPad = [&](std::vector<int> pad, int rank) {
            int padLen = pad.size();
            if (padLen == rank) {
                return pad;
            }
            std::vector<int> result;
            if (padLen > rank) {
                result.insert(result.end(), pad.begin(), pad.begin() + rank);
            } else {
                result = pad;
                result.insert(result.end(), rank - padLen, 0);
            }
            return result;
        };

        padBegin = correctPad(padBegin, dataRank);
        padEnd = correctPad(padEnd, dataRank);
        srcDimPad = getPaddedInputShape();
    } else {
        srcDimPad = srcDim;
    }
    dstDim = getChildEdgeAt(0)->getDims().ToSizeVector();

    // extract const buffer
    auto scalesLayer = getParentEdgesAtPort(SCALES_ID)[0]->getParent()->getCnnLayer();
    if (scalesLayer->type == "Const") {
        auto scalesBlob = dynamic_cast<TBlob<float>*>(scalesLayer->blobs["custom"].get());
        auto scalesData = scalesBlob->buffer().as<float*>();
        int scalesLen = getParentEdgeAt(SCALES_ID)->getDims()[0];
        scales.resize(scalesLen);
        for (int i = 0; i < scalesLen; i++) {
            scales[i] = scalesData[i];
        }
    } else {
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' only supports const 'scales' input.";
    }

    if (isAxesSpecified) {
        auto axesLayer = getParentEdgesAtPort(AXES_ID)[0]->getParent()->getCnnLayer();
        if (axesLayer->type == "Const") {
            auto axesBlob = dynamic_cast<TBlob<int>*>(axesLayer->blobs["custom"].get());
            auto axesData = axesBlob->buffer().as<int*>();
            int axesLen = getParentEdgeAt(AXES_ID)->getDims()[0];
            axes.resize(axesLen);
            for (int i = 0; i < axesLen; i++) {
                axes[i] = axesData[i];
            }
        } else {
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' only supports const 'axes' input.";
        }
    } else {
        int dataRank = srcDim.size();
        axes.resize(dataRank);
        for (int i = 0; i < dataRank; i++) {
            axes[i] = i;
        }
    }

    if (scales.size() != axes.size()) {
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() <<
        "' does not have the same number elements in scales as in axis.";
    }
}

void MKLDNNInterpolateNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    setPostOps(attr, true);

    Precision inputPrecision = getCnnLayer()->insData[DATA_ID].lock()->getPrecision();
    if ((inputPrecision != Precision::I8) && (inputPrecision != Precision::U8) && (inputPrecision != Precision::BF16)) {
        inputPrecision = Precision::FP32;
    }
    if ((inputPrecision == Precision::BF16) && !mayiuse(avx512_core_bf16)) {
        inputPrecision = Precision::FP32;
    }
    Precision outputPrecision = inputPrecision;

    if (!fusedWith.empty()) {
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputPrecision = lastFusedLayer->outData[0]->getPrecision();
        }
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);
    srcDataSize = MKLDNNExtensionUtils::sizeOfDataType(inputDataType);
    dstDataSize = MKLDNNExtensionUtils::sizeOfDataType(outputDataType);

    inputPrec = inputPrecision;
    outputPrec = outputPrecision;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    if (isAxesSpecified) {
        config.inConfs.resize(4);
    } else {
        config.inConfs.resize(3);
    }
    config.outConfs.resize(1);
    config.inConfs[DATA_ID].constant = false;
    config.inConfs[TARGET_SHAPE_ID].constant = false;
    config.inConfs[SCALES_ID].constant = false;
    config.outConfs[0].constant = false;
    config.inConfs[DATA_ID].inPlace = -1;
    config.inConfs[TARGET_SHAPE_ID].inPlace = -1;
    config.inConfs[SCALES_ID].inPlace = -1;
    config.outConfs[0].inPlace = -1;
    if (isAxesSpecified) {
        config.inConfs[AXES_ID].constant = false;
        config.inConfs[AXES_ID].inPlace = -1;
    }

    auto targetShapeType = MKLDNNExtensionUtils::IEPrecisionToDataType(Precision::I32);
    auto scalesType = MKLDNNExtensionUtils::IEPrecisionToDataType(Precision::FP32);
    auto axesType = MKLDNNExtensionUtils::IEPrecisionToDataType(Precision::I32);

    auto pushDesc = [&](memory::format dataFormat, impl_desc_type implDetail) {
        config.inConfs[DATA_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(DATA_ID)->getDims(), inputDataType, dataFormat);
        config.inConfs[TARGET_SHAPE_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(TARGET_SHAPE_ID)->getDims(), targetShapeType, memory::x);
        config.inConfs[SCALES_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(SCALES_ID)->getDims(), scalesType, memory::x);
        if (isAxesSpecified)
            config.inConfs[AXES_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(AXES_ID)->getDims(), axesType, memory::x);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, dataFormat);
        supportedPrimitiveDescriptors.push_back({config, implDetail, dataFormat});
    };

    if (mode != InterpolateMode::linear) {
        // blk and by_channel JIT kernel on sse42 or above machine
        if (mayiuse(cpu::sse42)) {
            if (getParentEdgeAt(DATA_ID)->getDims().ndims() == 4) {
                if (mayiuse(cpu::avx512_common)) {
                    pushDesc(memory::nhwc, jit_avx512);
                    pushDesc(memory::nChw16c, jit_avx512);
                } else if (mayiuse(cpu::avx2)) {
                    pushDesc(memory::nhwc, jit_avx2);
                    pushDesc(memory::nChw8c, jit_avx2);
                } else {
                    pushDesc(memory::nhwc, jit_sse42);
                    pushDesc(memory::nChw8c, jit_sse42);
                }
            } else if (getParentEdgeAt(DATA_ID)->getDims().ndims() == 5 && mode == InterpolateMode::nearest) {
                if (mayiuse(cpu::avx512_common)) {
                    pushDesc(memory::ndhwc, jit_avx512);
                    pushDesc(memory::nCdhw16c, jit_avx512);
                } else if (mayiuse(cpu::avx2)) {
                    pushDesc(memory::ndhwc, jit_avx2);
                    pushDesc(memory::nCdhw8c, jit_avx2);
                } else {
                    pushDesc(memory::ndhwc, jit_sse42);
                    pushDesc(memory::nCdhw8c, jit_sse42);
                }
            }
        }

        // planar for 1.ref on machine without sse42(if no sse42, canFuse() is false). 2.JIT kernel for f32 && avx2(gather).(with fuse)
        if (!mayiuse(cpu::sse42))
            pushDesc(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(DATA_ID)->getDims()), ref);

        if (mayiuse(cpu::avx2) && inputPrec == Precision::FP32) {
            pushDesc(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(DATA_ID)->getDims()), jit_avx2);
        }
    } else {
        pushDesc(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(DATA_ID)->getDims()), ref);
    }
}

void MKLDNNInterpolateNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();
    auto& tsMemPtr = getParentEdgeAt(TARGET_SHAPE_ID)->getMemoryPtr();
    auto& scaleMemPtr = getParentEdgeAt(SCALES_ID)->getMemoryPtr();
    if (getParentEdges().size() > 3) {
        auto &axesMemPtr = getParentEdgeAt(AXES_ID)->getMemoryPtr();
        if (!axesMemPtr || !axesMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' did not allocate axes memory";
    }
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' did not allocate destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' did not allocate input memory";
    if (!tsMemPtr || !tsMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' did not allocate target shape memory";
    if (!scaleMemPtr || !scaleMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' did not allocate scales memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' did not set preferable primitive descriptor";

    auto selectedPD = getSelectedPrimitiveDescriptor();
    Layout selected_layout = selectedPD->getConfig().inConfs[0].desc.getLayout();
    auto jcp = jit_interpolate_config_params();
    jcp.mode = mode;
    jcp.src_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[0].desc.getPrecision());
    jcp.dst_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().outConfs[0].desc.getPrecision());
    jcp.src_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.src_dt);
    jcp.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.dst_dt);
    jcp.indices_size = sizeof(int);
    size_t dimSize = dstDim.size();
    jcp.OW = dstDim[dimSize - 1];
    jcp.OH = dstDim[dimSize - 2];
    jcp.IW = srcDimPad[dimSize - 1];
    jcp.IH = srcDimPad[dimSize - 2];

    if (MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims()) == selected_layout) {
        jcp.layout = InterpolateLayoutType::planar;
    } else if ((selected_layout == NHWC) || (selected_layout == NDHWC)) {
        jcp.layout = InterpolateLayoutType::by_channel;
    } else {
        jcp.layout = InterpolateLayoutType::block;
    }

    if (mode == InterpolateMode::nearest || mode == InterpolateMode::linear_onnx || mode == InterpolateMode::cubic) {
        if (jcp.layout != InterpolateLayoutType::planar) {
            if (mayiuse(cpu::avx512_common)) {
                interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::avx512_common>(jcp, *attr.get()));
            } else if (mayiuse(cpu::avx2)) {
                interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::avx2>(jcp, *attr.get()));
            } else if (mayiuse(cpu::sse42)) {
                interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::sse42>(jcp, *attr.get()));
            }
        } else {
            // gather ISA(for planar JIT kernel) for avx2 and fp32
            if (mayiuse(cpu::avx2) && inputPrec == Precision::FP32) {
                interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::avx2>(jcp, *attr.get()));
            }
        }
    }

    // build indices table
    std::vector<float> dataScales = getScales();
    if (dimSize > 2 && (dataScales[0] != 1.f || dataScales[1] != 1.f)) {
        THROW_IE_EXCEPTION << "Interpolate layer only supports resize on spatial dimensions(depth, height and width)";
    }
    auto srcDimPad5d = to5Dim(srcDimPad);
    auto dstDim5d = to5Dim(dstDim);

    switch (mode) {
        case InterpolateMode::nearest: {
            buildTblNN(srcDimPad5d, dstDim5d, dataScales, jcp.layout);
            break;
        }
        case InterpolateMode::linear_onnx: {
            buildTblLinearOnnx(srcDimPad5d, dstDim5d, dataScales, jcp.layout);
            break;
        }
        case InterpolateMode::linear: {
            buildTblLinear(srcDimPad5d, dstDim5d, dataScales, LINEAR_KERNEL, antialias);
            break;
        }
        case InterpolateMode::cubic: {
            buildTblCubic(srcDimPad5d, dstDim5d, dataScales, cubeCoeff, jcp.layout);
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' does not support interpolate mode:" << mode;
            break;
        }
    }
}

int clipCoord(int pos, int length) {
    return std::max(static_cast<int>(0), std::min(pos, length - 1));
}

// index layout:
// d_0............d_OD-1, h_0..............h_OH-1, w_0................w_OW-1
void MKLDNNInterpolateNode::buildTblNN(SizeVector& srcDimPad5d, SizeVector& dstDim5d,
                                        std::vector<float>& dataScales, InterpolateLayoutType layout) {
    int dimSize = srcDim.size();
    float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.f;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    size_t ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    indexTable.resize(OD + OH + OW);
    bool isDDownsample = (fz < 1) ? true : false;
    bool isHDownsample = (fy < 1) ? true : false;
    bool isWDownsample = (fx < 1) ? true : false;
    for (int oz = 0; oz < OD; oz++) {
        float iz = coordTransToInput(oz, fz, ID, OD);
        indexTable[oz] = nearestRound(iz, isDDownsample);
        indexTable[oz] = clipCoord(indexTable[oz], ID);
    }
    for (int oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        indexTable[OD + oy] = nearestRound(iy, isHDownsample);
        indexTable[OD + oy] = clipCoord(indexTable[OD + oy], IH);
    }
    for (int ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        indexTable[OD + OH + ox] = nearestRound(ix, isWDownsample);
        indexTable[OD + OH + ox] = clipCoord(indexTable[OD + OH + ox], IW);
    }
}

void MKLDNNInterpolateNode::buildTblLinearOnnx(SizeVector& srcDimPad5d, SizeVector& dstDim5d,
                                                std::vector<float>& dataScales, InterpolateLayoutType layout) {
    int dimSize = srcDim.size();
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    int IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    int OH = dstDim5d[3], OW = dstDim5d[4];
    if (layout == InterpolateLayoutType::planar) {
        int eltInGrid = 4;
        int idxType = 2;
        int scratchLen = rnd_up(eltInGrid * OW * OH, 16);
        indexTable.resize(idxType * scratchLen);

        int *indexTopLeft = static_cast<int*>(&indexTable[0]);
        int *indexTopRight = static_cast<int*>(&indexTable[OW * OH]);
        int *indexBottomLeft = static_cast<int*>(&indexTable[2 * OW * OH]);
        int *indexBottomRight = static_cast<int*>(&indexTable[3 * OW * OH]);

        float *weightLeft = reinterpret_cast<float*>(&indexTable[scratchLen]);
        float *weightRight = reinterpret_cast<float*>(&indexTable[scratchLen + OW * OH]);
        float *weightTop = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW * OH]);
        float *weightBottom = reinterpret_cast<float*>(&indexTable[scratchLen + 3 * OW * OH]);

        for (int oy = 0; oy < OH; oy++) {
            float iy = coordTransToInput(oy, fy, IH, OH);
            iy = std::max(0.0f, std::min(iy, static_cast<float>(IH - 1)));
            int iyT = std::min(static_cast<int>(iy), IH - 1);
            int iyB = std::min(iyT + 1, IH - 1);

            float weightB = std::fabs(iy - iyT);
            float weightT = std::fabs(iy - iyB);
            if (iyT == iyB) {
                weightB = 0.5f;
                weightT = 0.5f;
            }
            int idxOy = oy * OW;
            for (int ox = 0; ox < OW; ox++) {
                float ix = coordTransToInput(ox, fx, IW, OW);
                ix = std::max(0.0f, std::min(ix, static_cast<float>(IW - 1)));
                int ixL = std::min(static_cast<int>(ix), IW - 1);
                int ixR = std::min(ixL + 1, IW - 1);

                float weightR = std::fabs(ix - ixL);
                float weightL = std::fabs(ix - ixR);
                if (ixL == ixR) {
                    weightR = 0.5f;
                    weightL = 0.5f;
                }
                int idxOyOx = idxOy + ox;
                indexTopLeft[idxOyOx] = (iyT * IW + ixL) * srcDataSize;
                indexTopRight[idxOyOx] = (iyT * IW + ixR) * srcDataSize;
                indexBottomLeft[idxOyOx] = (iyB * IW + ixL) * srcDataSize;
                indexBottomRight[idxOyOx] = (iyB * IW + ixR) * srcDataSize;
                weightLeft[idxOyOx] = weightL;
                weightRight[idxOyOx] = weightR;
                weightTop[idxOyOx] = weightT;
                weightBottom[idxOyOx] = weightB;
            }
        }
    } else {
        // left:OW right:OW Top:OH Bottom:OH
        size_t scratchLen = rnd_up(OW + OW + OH + OH, 16);
        int idxType = 2;
        indexTable.resize(idxType * scratchLen);
        int *indexLeft = static_cast<int*>(&indexTable[0]);
        int *indexRight = static_cast<int*>(&indexTable[OW]);
        int *indexTop = static_cast<int*>(&indexTable[2 * OW]);
        int *indexBottom = static_cast<int*>(&indexTable[2 * OW + OH]);

        float *weightLeft = reinterpret_cast<float*>(&indexTable[scratchLen]);
        float *weightRight = reinterpret_cast<float*>(&indexTable[scratchLen + OW]);
        float *weightTop = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW]);
        float *weightBottom = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW + OH]);

        for (int ox = 0; ox < OW; ox++) {
            float ix = coordTransToInput(ox, fx, IW, OW);
            ix = std::max(0.0f, std::min(ix, static_cast<float>(IW - 1)));
            indexLeft[ox] = std::min(static_cast<int>(ix), IW - 1);
            indexRight[ox] = std::min(indexLeft[ox] + 1, IW - 1);

            weightRight[ox] = std::fabs(ix - indexLeft[ox]);
            weightLeft[ox] = std::fabs(ix - indexRight[ox]);
            if (indexLeft[ox] == indexRight[ox]) {
                weightRight[ox] = 0.5f;
                weightLeft[ox] = 0.5f;
            }
        }

        for (int oy = 0; oy < OH; oy++) {
            float iy = coordTransToInput(oy, fy, IH, OH);
            iy = std::max(0.0f, std::min(iy, static_cast<float>(IH - 1)));
            indexTop[oy] = std::min(static_cast<int>(iy), IH - 1);
            indexBottom[oy] = std::min(indexTop[oy] + 1, IH - 1);

            weightBottom[oy] = std::fabs(iy - indexTop[oy]);
            weightTop[oy] = std::fabs(iy - indexBottom[oy]);
            if (indexTop[oy] == indexBottom[oy]) {
                weightBottom[oy] = 0.5f;
                weightTop[oy] = 0.5f;
            }
        }
    }
}

static inline float triangleCoeff(float x) {
    return (std::max)(0.0f, 1 - std::abs(x));
}

// table layout:
// wd .........wd, wh............wh, ww.............ww, id...........id, ih............ih, iw..............iw
//                        |                                                      |
//                   wh0.....wh_diameter                                    ih0.....ih_diameter
void MKLDNNInterpolateNode::buildTblLinear(SizeVector& srcDimPad5d, SizeVector& dstDim5d,
                                            std::vector<float>& dataScales, int kernel_width, bool antialias) {
    int dimSize = srcDim.size();
    float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.f;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    size_t ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    if (!(IW == OW && IH == OH && ID == OD)) {
        float ax = antialias ? fx : 1.0f;
        float ay = antialias ? fy : 1.0f;
        float az = antialias ? fz : 1.0f;

        int rx = (fx > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
        int ry = (fy > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
        int rz = (fz > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

        int diaOD = 2 * rz + 1;
        int diaOH = 2 * ry + 1;
        int diaOW = 2 * rx + 1;
        int sizeOD = OD * diaOD;
        int sizeOH = OH * diaOH;
        int sizeOW = OW * diaOW;
        indexTable.resize((sizeOD + sizeOH + sizeOW) * 2);
        float *weightTable = reinterpret_cast<float*>(&indexTable[0]);
        float *weightOD = static_cast<float*>(&weightTable[0]);
        float *weightOH = static_cast<float*>(&weightTable[sizeOD]);
        float *weightOW = static_cast<float*>(&weightTable[sizeOD + sizeOH]);

        int *idxTable = static_cast<int*>(&indexTable[sizeOD + sizeOH + sizeOW]);
        int *idxOD = static_cast<int*>(&idxTable[0]);
        int *idxOH = static_cast<int*>(&idxTable[sizeOD]);
        int *idxOW = static_cast<int*>(&idxTable[sizeOD + sizeOH]);

        for (int oz = 0; oz < OD; oz++) {
            float iz = coordTransToInput(oz, fz, ID, OD);
            int iz_r = static_cast<int>(std::round(iz));
            for (int r = iz_r - rz, i = 0; r <= iz_r + rz; r++, i++) {
                idxOD[oz * diaOD + i] = r;
                if (r < 0 || r >= static_cast<int>(ID)) {
                    weightOD[oz * diaOD + i] = 0.f;
                } else {
                    float dz = iz - r;
                    weightOD[oz * diaOD + i] = az * triangleCoeff(az * dz);
                }
            }
        }
        for (int oy = 0; oy < OH; oy++) {
            float iy = coordTransToInput(oy, fy, IH, OH);
            int iy_r = static_cast<int>(std::round(iy));
            for (int r = iy_r - ry, i = 0; r <= iy_r + ry; r++, i++) {
                idxOH[oy * diaOH + i] = r;
                if (r < 0 || r >= static_cast<int>(IH)) {
                    weightOH[oy * diaOH + i] = 0.f;
                } else {
                    float dy = iy - r;
                    weightOH[oy * diaOH + i] = ay * triangleCoeff(ay * dy);
                }
            }
        }
        for (int ox = 0; ox < OW; ox++) {
            float ix = coordTransToInput(ox, fx, IW, OW);
            int ix_r = static_cast<int>(std::round(ix));
            for (int r = ix_r - rx, i = 0; r <= ix_r + rx; r++, i++) {
                idxOW[ox * diaOW + i] = r;
                if (r < 0 || r >= static_cast<int>(IW)) {
                    weightOW[ox * diaOW + i] = 0.f;
                } else {
                    float dx = ix - r;
                    weightOW[ox * diaOW + i] = ax * triangleCoeff(ax * dx);
                }
            }
        }
    }
}

std::vector<float> MKLDNNInterpolateNode::getCubicCoeffs(float mantissa, float a) {
    float m = std::fabs(mantissa);
    std::vector<float> coeffs(4, 0.f);

    coeffs[0] = a * (m - 1.0) * (m - 1.0) * m;
    coeffs[1] = ((a + 2.0) * m - (a + 3.0)) * m * m + 1.0;
    coeffs[2] = (((-a - 2.0) * m + (2.0 * a + 3.0)) * m - a) * m;
    coeffs[3] = -a * m * m * (m - 1.0);
    return coeffs;
}

// table layout:
// OW      OW         OW         OW         OW          OH       OH           OH           OH           OH
// x_idx   x_weight0  x_weight1  x_weight2  x_weight3   y_idx    y_weight0    y_weight1    y_weight2    y_weight3
void MKLDNNInterpolateNode::buildTblCubic(SizeVector& srcDimPad5d, SizeVector& dstDim5d, std::vector<float>& dataScales,
                                        float cubicCoeff, InterpolateLayoutType layout) {
    int dimSize = srcDim.size();
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    int IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    int OH = dstDim5d[3], OW = dstDim5d[4];

    // idxNum for index, CUBIC_GRID_LEN for weight
    const int idxNum = 1;
    size_t idxWeightSize = (CUBIC_GRID_LEN + idxNum) * OW + (CUBIC_GRID_LEN + idxNum) * OH;
    if (layout != InterpolateLayoutType::planar) {
        indexTable.resize(idxWeightSize);
    } else {
        size_t sequenceSize = 2 * OH * OW;
        indexTable.resize(idxWeightSize + sequenceSize);
    }

    int tblAdvance = 0;
    int *xOrigin = static_cast<int*>(&indexTable[tblAdvance]);
    tblAdvance += OW;
    float *xFactor = reinterpret_cast<float*>(&indexTable[tblAdvance]);
    for (int ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        int ix_r = static_cast<int>(std::floor(ix));
        xOrigin[ox] = ix_r;
        float m = ix - ix_r;
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        xFactor[CUBIC_GRID_LEN * ox] = coffes[0];
        xFactor[CUBIC_GRID_LEN * ox + 1] = coffes[1];
        xFactor[CUBIC_GRID_LEN * ox + 2] = coffes[2];
        xFactor[CUBIC_GRID_LEN * ox + 3] = coffes[3];
    }

    tblAdvance += CUBIC_GRID_LEN * OW;
    int *yOrigin = static_cast<int*>(&indexTable[tblAdvance]);
    tblAdvance += OH;
    float *yFactor = reinterpret_cast<float*>(&indexTable[tblAdvance]);
    for (int oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        int iy_r = static_cast<int>(std::floor(iy));
        yOrigin[oy] = iy_r;
        float m = iy - iy_r;
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        yFactor[CUBIC_GRID_LEN * oy] = coffes[0];
        yFactor[CUBIC_GRID_LEN * oy + 1] = coffes[1];
        yFactor[CUBIC_GRID_LEN * oy + 2] = coffes[2];
        yFactor[CUBIC_GRID_LEN * oy + 3] = coffes[3];
    }

    if (layout == InterpolateLayoutType::planar) {
        tblAdvance += CUBIC_GRID_LEN * OH;
        int *sequenceOH = static_cast<int*>(&indexTable[tblAdvance]);
        tblAdvance += OH * OW;
        int *sequenceOW = static_cast<int*>(&indexTable[tblAdvance]);
        for (int h = 0; h < OH; ++h) {
            int offset = h * OW;
            for (int w = 0; w < OW; ++w) {
                sequenceOH[offset + w] = h * sizeof(int);
                sequenceOW[offset + w] = w * sizeof(int);
            }
        }
    }
}

void MKLDNNInterpolateNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
    int blob_idx = 0;
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode) {
            quantizeNode->appendPostOps(ops);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops);
            continue;
        }

        THROW_IE_EXCEPTION << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

SizeVector MKLDNNInterpolateNode::getPaddedInputShape() {
    SizeVector paddedShape;
    int dataRank = srcDim.size();
    for (int i = 0; i < dataRank; i++) {
        paddedShape.push_back(srcDim[i] + padBegin[i] + padEnd[i]);
    }
    return paddedShape;
}

// get scales of data rank size
// if "scale" version: set scales with input scales, 1.f for other dims not in axis
// if "size" version: scales = shape[target] / shape[input].pad, 1.f for other dims not in axis
// scales is a required input, but should not use input scales when "size" case, which may added eps that lead to inaccurate result, recalculate scales instead.
std::vector<float> MKLDNNInterpolateNode::getScales() {
    int dataRank = srcDim.size();
    std::vector<float> fullScales(dataRank, 1.f);
    int axesRank = axes.size();
    for (int i = 0; i < axesRank; i++) {
        int axis = axes[i];
        fullScales[axis] = (shapeInferMode == "scales") ? scales[i] : static_cast<float>(dstDim[axis]) / static_cast<float>(srcDimPad[axis]);
    }
    return fullScales;
}

void MKLDNNInterpolateNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();

    uint8_t *dst_data = reinterpret_cast<uint8_t*>(dstMemPtr->GetData()) +
            dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding * dstDataSize;
    uint8_t *src_data_origin = reinterpret_cast<uint8_t*>(srcMemPtr->GetData()) +
            srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding * srcDataSize;

    size_t dimSize = srcDim.size();
    SizeVector srcDimPad = getPaddedInputShape();

    auto srcDim5d = to5Dim(srcDim);
    auto srcDimPad5d = to5Dim(srcDimPad);
    auto dstDim5d = to5Dim(dstDim);

    InterpolateLayoutType layout;
    Layout selected_layout = getParentEdgeAt(DATA_ID)->getDesc().getLayout();
    if (MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims()) == selected_layout) {
        layout = InterpolateLayoutType::planar;
    } else if ((selected_layout == NHWC) || (selected_layout == NDHWC)) {
        layout = InterpolateLayoutType::by_channel;
    } else {
        layout = InterpolateLayoutType::block;
    }

    uint8_t *src_data = nullptr;
    std::vector<uint8_t> srcPadded;
    if (hasPad) {
        int padB0 = (dimSize > 2) ? padBegin[0] : 0;
        int padB1 = (dimSize > 2) ? padBegin[1] : 0;
        int padB2 = (dimSize == 5) ? padBegin[dimSize - 3] : 0;
        int padB3 = padBegin[dimSize - 2];
        int padB4 = padBegin[dimSize - 1];

        SizeVector inShapeBlock = getBlockND(srcDim5d);
        SizeVector inShapePadBlock = getBlockND(srcDimPad5d);

        if (layout == InterpolateLayoutType::planar) {
            srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
            uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);
            parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], [&](int n, int c, int d, int h) {
                uint8_t *src = src_data_origin + (inShapeBlock[1] * n + inShapeBlock[2] * c + inShapeBlock[3] * d + inShapeBlock[4] * h) * srcDataSize;
                uint8_t *srcPad = src_data_pad + (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) +
                               inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4) * srcDataSize;
                cpu_memcpy(srcPad, src, srcDim5d[4] * srcDataSize);
            });
            src_data = src_data_pad;
        } else if (layout == InterpolateLayoutType::by_channel) {
            srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
            uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);
            parallel_for4d(srcDim5d[0], srcDim5d[2], srcDim5d[3], srcDim5d[4], [&](int n, int d, int h, int w) {
                uint8_t *src = src_data_origin + (inShapeBlock[1] * n +
                                (inShapeBlock[3] * d + inShapeBlock[4] * h + inShapeBlock[5] * w) * srcDim5d[1]) * srcDataSize;
                uint8_t *srcPad = src_data_pad + (inShapePadBlock[1] * (n + padB0) + (inShapePadBlock[3] * (d + padB2) +
                                inShapePadBlock[4] * (h + padB3) + inShapePadBlock[5] * (w + padB4)) * srcDimPad5d[1] + padB1) * srcDataSize;
                cpu_memcpy(srcPad, src, srcDim5d[1] * srcDataSize);
            });
            src_data = src_data_pad;
        } else if (layout == InterpolateLayoutType::block) {
            size_t blkSize = mayiuse(cpu::avx512_common) ? 16 : 8;
            size_t CB = div_up(srcDimPad5d[1], blkSize);
            size_t eltsTotal = srcDimPad5d[0] * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize;
            srcPadded.resize(eltsTotal * srcDataSize, 0x0);
            uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);
            if ((srcDim5d[0] != srcDimPad5d[0]) || (srcDim5d[1] != srcDimPad5d[1])) {
                THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() <<
                "' does not support padding on batch and channel dimensions";
            }
            parallel_for5d(srcDim5d[0], CB, srcDim5d[2], srcDim5d[3], srcDim5d[4], [&](int n, int cb, int d, int h, int w) {
                uint8_t *src = src_data_origin + (n * CB * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize
                                               + (cb * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize
                                               + (d * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize
                                               + (h * srcDim5d[4] * blkSize) * srcDataSize
                                               + (w * blkSize) * srcDataSize;
                uint8_t *srcPad = src_data_pad + (n * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize
                                               + (cb * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize
                                               + ((d + padB2) * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize
                                               + ((h + padB3) * srcDimPad5d[4] * blkSize) * srcDataSize
                                               + ((w + padB4) * blkSize) * srcDataSize;
                cpu_memcpy(srcPad, src, blkSize * srcDataSize);
            });
            src_data = src_data_pad;
        }
    } else {
        src_data = src_data_origin;
    }

    size_t N = srcDimPad5d[0], C = srcDimPad5d[1], ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];
    std::vector<float> dataScales = getScales();
    if (dimSize > 2 && (dataScales[0] != 1.f || dataScales[1] != 1.f)) {
        THROW_IE_EXCEPTION << "Interpolate layer only supports resize on spatial dimensions(depth, height and width)";
    }

    switch (mode) {
        case InterpolateMode::nearest: {
            if (interpolateKernel) {
                if (layout == InterpolateLayoutType::planar) {
                    NNPlanar(src_data, dst_data, N, C, ID, IH, IW, OD, OH, OW);
                } else {
                    NNCGathered(src_data, dst_data, N, C, ID, IH, IW, OD, OH, OW);
                }
            } else {
                NNRef(src_data, dst_data, N, C, ID, IH, IW, OD, OH, OW);
            }
            break;
        }
        case InterpolateMode::linear_onnx: {
            if (interpolateKernel) {
                if (layout == InterpolateLayoutType::planar) {
                    linearOnnxPlanar(src_data, dst_data, N, C, IH, IW, OH, OW);
                } else {
                    linearOnnxCGathered(src_data, dst_data, N, C, IH, IW, OH, OW);
                }
            } else {
                linearOnnxRef(src_data, dst_data, N, C, IH, IW, OH, OW);
            }
            break;
        }
        case InterpolateMode::cubic: {
            if (interpolateKernel) {
                if (layout == InterpolateLayoutType::planar) {
                    cubicPlanar(src_data, dst_data, N, C, IH, IW, OH, OW);
                } else {
                    cubicCGathered(src_data, dst_data, N, C, IH, IW, OH, OW);
                }
            } else {
                cubicRef(src_data, dst_data, N, C, IH, IW, OH, OW);
            }
            break;
        }
        case InterpolateMode::linear: {
            float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.f;
            float fy = dataScales[dimSize - 2];
            float fx = dataScales[dimSize - 1];

            bool isDownsample = (fx < 1.f) || (fy < 1.f) || (fz < 1.f);
            int kernel_width = 2;
            linearInterpolation(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer has unsupported interpolate mode: " << mode;
        }
    }
}

// for ndhwc and nCdhw8c[16c]
// input may be f32/bf16/int8, fused->output varies
void MKLDNNInterpolateNode::NNCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    int *index_d = static_cast<int*>(&indexTable[0]);
    int *index_h = static_cast<int*>(&indexTable[OD]);
    int *index_w = static_cast<int*>(&indexTable[OD + OH]);

    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();
    bool is_nhwc = (layout == NHWC || layout == NDHWC) ? true : false;

    for (int b = 0; b < B; b++) {
        if (is_nhwc) {
            const uint8_t *in_ptr = in_ptr_ + (IW * IH * ID * C * b) * srcDataSize;
            uint8_t *out_ptr = out_ptr_ + (OW * OH * OD * C * b) * dstDataSize;
            std::vector<int> index_w_kernel(OW);
            for (int ox = 0; ox < OW; ox++) {
                index_w_kernel[ox] = index_w[ox] * C * srcDataSize;
            }
            parallel_for2d(OD, OH, [&](size_t d, size_t h) {
                // kernel for C * OW
                uint8_t *out_ptr_dh = out_ptr + (C * OW * OH * d + C * OW * h) * dstDataSize;
                const uint8_t *in_ptr_dh = in_ptr + (C * IW * IH * index_d[d] + C * IW * index_h[h]) * srcDataSize;
                auto arg = jit_interpolate_call_args();
                arg.dst = out_ptr_dh;
                arg.src_ptr[0] = in_ptr_dh;
                arg.index = static_cast<int*>(&(index_w_kernel[0]));
                arg.work_amount = C;
                arg.oc_off = 0;
                (*interpolateKernel)(&arg);
            });
        } else {  // for blk
            int blk_size = mayiuse(cpu::avx512_common) ? 16 : 8;
            int CB = div_up(C, blk_size);
            const uint8_t *in_ptr = in_ptr_ + (IW * IH * ID * CB * blk_size * b) * srcDataSize;
            uint8_t *out_ptr = out_ptr_ + (OW * OH * OD * CB * blk_size * b) * dstDataSize;
            std::vector<int> index_w_kernel(OW);
            for (int ox = 0; ox < OW; ox++) {
                index_w_kernel[ox] = index_w[ox] * blk_size * srcDataSize;
            }
            parallel_for2d(CB, OD, [&](size_t cb, size_t d) {
                uint8_t *out_ptr_cbd = out_ptr + (blk_size * OW * OH * OD * cb + blk_size * OW * OH * d) * dstDataSize;
                const uint8_t *in_ptr_cbd = in_ptr + (blk_size * IW * IH * ID * cb + blk_size * IW * IH * index_d[d]) * srcDataSize;
                auto arg = jit_interpolate_call_args();
                for (int h = 0; h < OH; h++) {  // kernel for blk_size * OW
                    arg.dst = out_ptr_cbd + blk_size * OW * h * dstDataSize;
                    arg.src_ptr[0] = in_ptr_cbd + blk_size * IW * index_h[h] * srcDataSize;
                    arg.index = static_cast<int*>(&(index_w_kernel[0]));
                    arg.work_amount = static_cast<size_t>(OW);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    (*interpolateKernel)(&arg);
                }
            });
        }
    }  // batch end
}

void MKLDNNInterpolateNode::NNPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    int *index_d = static_cast<int*>(&indexTable[0]);
    int *index_h = static_cast<int*>(&indexTable[OD]);
    int *index_w = static_cast<int*>(&indexTable[OD + OH]);

    std::vector<int> index_kernel(OH + OW);
    // index_h * IW * srcDataSize to reduce and simplify redundant compute
    for (int oh = 0; oh < OH; oh++) {
        index_kernel[oh] = index_h[oh] * IW * srcDataSize;
    }
    // index_w * srcDataSize
    for (int ow = 0; ow < OW; ow++) {
        index_kernel[OH + ow] = index_w[ow] * srcDataSize;
    }

    parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        const uint8_t *in_ptr = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]) * srcDataSize;
        uint8_t *out_ptr = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od) * dstDataSize;

        auto arg = jit_interpolate_call_args();
        arg.src_ptr[0] = in_ptr;
        arg.dst = out_ptr;
        arg.index = static_cast<int*>(&index_kernel[0]);  // need index_h and index_w in kernel, it's in continous memory so one param
        arg.oc_off = static_cast<size_t>(c * sizeof(float));
        // work_amount is OH(out loop) and OW(inner loop), can get in kernel from jcp.
        (*interpolateKernel)(&arg);
    });
}

void MKLDNNInterpolateNode::NNRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    int *index_d = static_cast<int*>(&indexTable[0]);
    int *index_h = static_cast<int*>(&indexTable[OD]);
    int *index_w = static_cast<int*>(&indexTable[OD + OH]);

    parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        const uint8_t *in_ptr = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]) * srcDataSize;
        uint8_t *out_ptr = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od) * dstDataSize;
        for (int oh = 0; oh < OH; oh++) {
            const uint8_t *in_ptr_h = in_ptr + (IW * index_h[oh]) * srcDataSize;
            uint8_t *out_ptr_h = out_ptr + (OW * oh) * dstDataSize;
            for (int ow = 0; ow < OW; ow++) {
                float dstValue = getValue(in_ptr_h, index_w[ow] * srcDataSize, inputPrec);
                setValue(out_ptr_h, ow * dstDataSize, dstValue, outputPrec);
            }
        }
    });
}

void MKLDNNInterpolateNode::linearOnnxPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW) {
    int *index = static_cast<int*>(&indexTable[0]);
    int eltInGrid = 4;
    int scratchLen = rnd_up(eltInGrid * OW * OH, 16);
    float *weight = reinterpret_cast<float*>(&indexTable[scratchLen]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        uint8_t *out_ptr_nc = out_ptr_ + (OH * OW * C * b + OH * OW * c) * dstDataSize;
        const uint8_t *in_ptr_nc = in_ptr_ + (IH * IW * C * b + IH * IW * c) * srcDataSize;
        auto arg = jit_interpolate_call_args();
        arg.src_ptr[0] = in_ptr_nc;
        arg.index = static_cast<int*>(&index[0]);
        arg.weight_ptr[0] = static_cast<float*>(&weight[0]);
        arg.dst = out_ptr_nc;
        arg.work_amount = OW * OH;
        arg.oc_off = static_cast<size_t>(c * sizeof(float));
        (*interpolateKernel)(&arg);
    });
}

void MKLDNNInterpolateNode::linearOnnxCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW) {
    // left:OW right:OW Top:OH Bottom:OH
    size_t scratchLen = rnd_up(OW + OW + OH + OH, 16);
    int *indexLeft = static_cast<int*>(&indexTable[0]);
    int *indexRight = static_cast<int*>(&indexTable[OW]);
    int *indexTop = static_cast<int*>(&indexTable[2 * OW]);
    int *indexBottom = static_cast<int*>(&indexTable[2 * OW + OH]);

    float *weightLeft = reinterpret_cast<float*>(&indexTable[scratchLen]);
    float *weightRight = reinterpret_cast<float*>(&indexTable[scratchLen + OW]);
    float *weightTop = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW]);
    float *weightBottom = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW + OH]);

    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();
    bool isByChannel = (layout == NHWC) ? true : false;

    int blkSize = mayiuse(cpu::avx512_common) ? 16 : 8;
    int CB = div_up(C, blkSize);
    int CSize = isByChannel ? C : blkSize * CB;
    int CGatherLen = isByChannel ? C : blkSize;
    int workAmount = isByChannel ? C : CB;
    parallel_for2d(B, OH, [&](size_t b, size_t h) {
        uint8_t *out_ptr_nh = out_ptr_ + (OH * OW * CSize * b + OW * CGatherLen * h) * dstDataSize;
        const uint8_t *in_ptr_n = in_ptr_ + (IH * IW * CSize * b) * srcDataSize;
        const uint8_t *in_ptr_nh_t = in_ptr_n + (indexTop[h] * IW * CGatherLen) * srcDataSize;
        const uint8_t *in_ptr_nh_b = in_ptr_n + (indexBottom[h] * IW * CGatherLen) * srcDataSize;
        auto arg = jit_interpolate_call_args();
        for (int w = 0; w < OW; ++w) {
            uint8_t *out_ptr_nhw = out_ptr_nh + CGatherLen * w * dstDataSize;
            arg.src_ptr[0] = in_ptr_nh_t + (indexLeft[w] * CGatherLen) * srcDataSize;
            arg.src_ptr[1] = in_ptr_nh_t + (indexRight[w] * CGatherLen) * srcDataSize;
            arg.src_ptr[2] = in_ptr_nh_b + (indexLeft[w] * CGatherLen) * srcDataSize;
            arg.src_ptr[3] = in_ptr_nh_b + (indexRight[w] * CGatherLen) * srcDataSize;
            arg.weight_ptr[0] = static_cast<float*>(&weightLeft[w]);
            arg.weight_ptr[1] = static_cast<float*>(&weightRight[w]);
            arg.weight_ptr[2] = static_cast<float*>(&weightTop[h]);
            arg.weight_ptr[3] = static_cast<float*>(&weightBottom[h]);
            arg.dst = out_ptr_nhw;
            arg.work_amount = workAmount;
            arg.oc_off = 0;
            (*interpolateKernel)(&arg);
        }
    });
}

void MKLDNNInterpolateNode::linearOnnxRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW) {
    int eltInGrid = 4;
    int scratchLen = rnd_up(eltInGrid * OW * OH, 16);

    int *indexTopLeft = static_cast<int*>(&indexTable[0]);
    int *indexTopRight = static_cast<int*>(&indexTable[OW * OH]);
    int *indexBottomLeft = static_cast<int*>(&indexTable[2 * OW * OH]);
    int *indexBottomRight = static_cast<int*>(&indexTable[3 * OW * OH]);

    float *weightLeft = reinterpret_cast<float*>(&indexTable[scratchLen]);
    float *weightRight = reinterpret_cast<float*>(&indexTable[scratchLen + OW * OH]);
    float *weightTop = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW * OH]);
    float *weightBottom = reinterpret_cast<float*>(&indexTable[scratchLen + 3 * OW * OH]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        uint8_t *out_ptr_nc = out_ptr_ + (OH * OW * C * b + OH * OW * c) * dstDataSize;
        const uint8_t *in_ptr_nc = in_ptr_ + (IH * IW * C * b + IH * IW * c) * srcDataSize;
        for (int i = 0; i < OH * OW; i++) {
            float srcTL = getValue(in_ptr_nc,  indexTopLeft[i], inputPrec);
            float srcTR = getValue(in_ptr_nc,  indexTopRight[i], inputPrec);
            float srcBL = getValue(in_ptr_nc,  indexBottomLeft[i], inputPrec);
            float srcBR = getValue(in_ptr_nc,  indexBottomRight[i], inputPrec);

            float dstValue = srcTL * weightTop[i] * weightLeft[i] + srcTR * weightTop[i] * weightRight[i] +
                        srcBL * weightBottom[i] * weightLeft[i] + srcBR * weightBottom[i] * weightRight[i];

            setValue(out_ptr_nc, i * dstDataSize, dstValue, outputPrec);
        }
    });
}

void MKLDNNInterpolateNode::linearInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias) {
    if (IW == OW && IH == OH && ID == OD) {
        size_t spatialDimSize = IW * IH * ID;
        if (fusedWith.empty() && inputPrec == outputPrec) {
            size_t size = B * C * spatialDimSize * srcDataSize;
            cpu_memcpy(out_ptr_, in_ptr_, size);
        } else {
            parallel_for2d(B, C, [&](size_t b, size_t c) {
                const uint8_t *in_ptr_nc = in_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * srcDataSize;
                uint8_t *out_ptr_nc = out_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * dstDataSize;
                for (size_t i = 0; i < spatialDimSize; i++) {
                    float dstValue = getValue(in_ptr_nc, i * srcDataSize, inputPrec);
                    setValue(out_ptr_nc, i * dstDataSize, dstValue, outputPrec);
                }
            });
        }
        return;
    }

    float ax = antialias ? fx : 1.0f;
    float ay = antialias ? fy : 1.0f;
    float az = antialias ? fz : 1.0f;

    int rx = (fx > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
    int ry = (fy > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
    int rz = (fz > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

    int diaOD = 2 * rz + 1;
    int diaOH = 2 * ry + 1;
    int diaOW = 2 * rx + 1;
    int sizeOD = OD * diaOD;
    int sizeOH = OH * diaOH;
    int sizeOW = OW * diaOW;

    float *weightTable = reinterpret_cast<float*>(&indexTable[0]);
    float *weightOD = static_cast<float*>(&weightTable[0]);
    float *weightOH = static_cast<float*>(&weightTable[sizeOD]);
    float *weightOW = static_cast<float*>(&weightTable[sizeOD + sizeOH]);

    int *idxTable = static_cast<int*>(&indexTable[sizeOD + sizeOH + sizeOW]);
    int *idxOD = static_cast<int*>(&idxTable[0]);
    int *idxOH = static_cast<int*>(&idxTable[sizeOD]);
    int *idxOW = static_cast<int*>(&idxTable[sizeOD + sizeOH]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        const uint8_t *in_ptr_nc = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c) * srcDataSize;
        uint8_t *out_ptr_nc = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c) * dstDataSize;
        for (size_t oz = 0; oz < OD; oz++) {
            uint8_t *out_ptr_ncd = out_ptr_nc + (OW * OH * oz) * dstDataSize;
            for (size_t oy = 0; oy < OH; oy++) {
                uint8_t *out_ptr_ncdh = out_ptr_ncd + (OW * oy) * dstDataSize;
                for (size_t ox = 0; ox < OW; ox++) {
                    float sum = 0.f;
                    float wsum = 0.f;

                    // this comment explains the original algo.
                    // for (int z = iz_r - rz; z <= iz_r + rz; z++) {
                    //    for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                    //        for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                    //            bool is_continue =  z < 0                     ||
                    //                                y < 0                     ||
                    //                                x < 0                     ||
                    //                                z >= static_cast<int>(ID) ||
                    //                                y >= static_cast<int>(IH) ||
                    //                                x >= static_cast<int>(IW);
                    //            if (is_continue)
                    //                continue;

                    //            float dx = ix - x;
                    //            float dy = iy - y;
                    //            float dz = iz - z;

                    //            float w = ax * triangleCoeff(ax * dx) *
                    //                      ay * triangleCoeff(ay * dy) *
                    //                      az * triangleCoeff(az * dz);

                    //            sum += w * getValue(in_ptr_nc, (z * IH * IW + y * IW + x) * srcDataSize, inputPrec);
                    //            wsum += w;
                    //        }
                    //    }
                    //}

                    for (int iz = 0; iz < diaOD; iz++) {
                        if (weightOD[oz * diaOD + iz] == 0.f)
                            continue;
                        for (int iy = 0; iy < diaOH; iy++) {
                            if (weightOH[oy * diaOH + iy] == 0.f) {
                                continue;
                            }
                            for (int ix = 0; ix < diaOW; ix++) {
                                if (weightOW[ox * diaOW + ix] == 0.f) {
                                    continue;
                                }
                                float w = weightOD[oz * diaOD + iz] * weightOH[oy * diaOH + iy] * weightOW[ox * diaOW + ix];
                                float value = getValue(in_ptr_nc,
                                    (idxOD[oz * diaOD + iz] * IH * IW + idxOH[oy * diaOH + iy] * IW + idxOW[ox * diaOW + ix]) * srcDataSize, inputPrec);

                                sum += w * value;
                                wsum += w;
                            }
                        }
                    }

                    if (!wsum) {
                        setValue(out_ptr_ncdh, ox * dstDataSize, 0.f, outputPrec);
                    } else {
                        float dst_value = sum / wsum;
                        setValue(out_ptr_ncdh, ox * dstDataSize, dst_value, outputPrec);
                    }
                }
            }
        }
    });
}

void MKLDNNInterpolateNode::cubicCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW) {
    const int idxNum = 1;
    int *xOrigin = static_cast<int*>(&indexTable[0]);
    float *xFactor = reinterpret_cast<float*>(&indexTable[OW]);
    int *yOrigin = static_cast<int*>(&indexTable[(CUBIC_GRID_LEN + idxNum) * OW]);
    float *yFactor = reinterpret_cast<float*>(&indexTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);

    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();
    bool isByChannel = (layout == NHWC) ? true : false;

    int blkSize = mayiuse(cpu::avx512_common) ? 16 : 8;
    int CB = div_up(C, blkSize);
    int CSize = isByChannel ? C : blkSize * CB;
    int CGatherLen = isByChannel ? C : blkSize;
    int workAmount = isByChannel ? C : CB;

    parallel_for3d(B, OH, OW, [&](size_t b, size_t h, size_t w) {
        uint8_t *out_ptr_nhw = out_ptr_ + (OH * OW * CSize * b + OW * CGatherLen * h + CGatherLen * w) * dstDataSize;
        const uint8_t *in_ptr_n = in_ptr_ + (IH * IW * CSize * b) * srcDataSize;

        std::vector<int> kernelIndex(CUBIC_GRID_LEN * CUBIC_GRID_LEN);  // 16 address offset to src(batch) or src(CB)
        int iy = yOrigin[h];
        int ix = xOrigin[w];
        for (int y = iy - 1, i = 0; y <= iy + 2; y++, i++) {
            int yInRange = std::max(0, std::min(y, IH - 1));
            yInRange = yInRange * CGatherLen * IW * srcDataSize;
            for (int x = ix - 1, j = 0; x <= ix + 2; x++, j++) {
                int xInRange = std::max(0, std::min(x, IW - 1));
                xInRange = yInRange + xInRange * CGatherLen * srcDataSize;
                kernelIndex[i * CUBIC_GRID_LEN + j] = xInRange;
            }
        }
        auto arg = jit_interpolate_call_args();
            arg.dst = out_ptr_nhw;
            arg.src_ptr[0] = in_ptr_n;
            arg.index = static_cast<int*>(&kernelIndex[0]);
            // 0 for weight_W, 1 for weight_H
            arg.weight_ptr[0] = static_cast<float*>(&xFactor[w * CUBIC_GRID_LEN]);
            arg.weight_ptr[1] = static_cast<float*>(&yFactor[h * CUBIC_GRID_LEN]);

            // for by channel, src + step, dst + step, process next step on continuous memory
            // for blk, src + IW*IH*blkSize, dst + OW*OH*blkSize, process the blkSize on next CB
            arg.work_amount = workAmount;
            arg.oc_off = 0;
            (*interpolateKernel)(&arg);
    });
}

void MKLDNNInterpolateNode::cubicPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW) {
    const int idxNum = 1;
    int tblAdvance = 0;
    int *xOrigin = static_cast<int*>(&indexTable[tblAdvance]);
    tblAdvance += OW;
    float *xFactor = reinterpret_cast<float*>(&indexTable[tblAdvance]);
    tblAdvance += CUBIC_GRID_LEN * OW;
    int *yOrigin = static_cast<int*>(&indexTable[tblAdvance]);
    tblAdvance += OH;
    float *yFactor = reinterpret_cast<float*>(&indexTable[tblAdvance]);

    tblAdvance += CUBIC_GRID_LEN * OH;
    int *sequenceOH = static_cast<int*>(&indexTable[tblAdvance]);
    tblAdvance += OW * OH;
    int *sequenceOW = static_cast<int*>(&indexTable[tblAdvance]);

    parallel_for2d(B, C, [&](size_t n, size_t c) {
        const uint8_t *in_ptr_nc = in_ptr_ + (IW * IH * C * n + IW * IH * c) * srcDataSize;
        uint8_t *out_ptr_nc = out_ptr_ + (OW * OH * C * n + OW * OH * c) * dstDataSize;

        auto arg = jit_interpolate_call_args();
        arg.dst = out_ptr_nc;
        arg.src_ptr[0] = in_ptr_nc;
        arg.index = xOrigin;
        arg.src_ptr[1] = yOrigin;
        arg.src_ptr[2] = static_cast<int*>(&sequenceOH[0]);
        arg.src_ptr[3] = static_cast<int*>(&sequenceOW[0]);
        arg.weight_ptr[0] = xFactor;
        arg.weight_ptr[1] = yFactor;
        arg.work_amount = static_cast<size_t>(OW * OH);
        arg.oc_off = static_cast<size_t>(c * sizeof(float));
        (*interpolateKernel)(&arg);
    });
}

void MKLDNNInterpolateNode::cubicRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW) {
    const int idxNum = 1;
    int *xOrigin = static_cast<int*>(&indexTable[0]);
    float *xFactor = reinterpret_cast<float*>(&indexTable[OW]);
    int *yOrigin = static_cast<int*>(&indexTable[(CUBIC_GRID_LEN + idxNum) * OW]);
    float *yFactor = reinterpret_cast<float*>(&indexTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);

    parallel_for4d(B, C, OH, OW, [&](size_t n, size_t c, size_t oy, size_t ox) {
        const uint8_t *in_ptr_nc = in_ptr_ + (IW * IH * C * n + IW * IH * c) * srcDataSize;
        uint8_t *out_ptr_nc = out_ptr_ + (OW * OH * C * n + OW * OH * c) * dstDataSize;

        int iy = yOrigin[oy];
        int ix = xOrigin[ox];

        float retY = 0.f;
        for (int y = iy - 1, i = 0; y <= iy + 2; y++, i++) {
            int yInRange = std::max(0, std::min(y, IH - 1));
            const uint8_t *in_ptr_nch = in_ptr_nc + IW * yInRange * srcDataSize;
            float retX = 0.f;
            for (int x = ix - 1, j = 0; x <= ix + 2; x++, j++) {
                int xInRange = std::max(0, std::min(x, IW - 1));
                retX += xFactor[ox * CUBIC_GRID_LEN + j] * getValue(in_ptr_nch, xInRange * srcDataSize, inputPrec);
            }
            retY += yFactor[oy * CUBIC_GRID_LEN + i] * retX;
        }

        setValue(out_ptr_nc, (oy * OW + ox) * dstDataSize, retY, outputPrec);
    });
}

float MKLDNNInterpolateNode::getValue(const uint8_t *base, size_t offset, InferenceEngine::Precision prec) {
    const uint8_t *baseOffset = base + offset;
    switch (prec) {
        case Precision::U8: {
            return static_cast<float>(*baseOffset);
            break;
        }
        case Precision::I8: {
            const int8_t *valuePtr = reinterpret_cast<const int8_t *>(baseOffset);
            return static_cast<float>(*valuePtr);
            break;
        }
        case Precision::BF16: {
            const uint16_t *valuePtr = reinterpret_cast<const uint16_t *>(baseOffset);
            return ngraph::bfloat16::from_bits(*valuePtr);
            break;
        }
        case Precision::FP32: {
            const float *valuePtr = reinterpret_cast<const float *>(baseOffset);
            return *valuePtr;
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer does not support precision: " << prec;
            break;
        }
    }
}

void MKLDNNInterpolateNode::setValue(uint8_t *base, size_t offset, float value, InferenceEngine::Precision prec) {
    uint8_t *baseOffset = base + offset;
    switch (prec) {
        case Precision::U8: {
            uint8_t data = static_cast<uint8_t>(value < 0 ? 0 : value);
            std::memcpy(baseOffset, &data, 1);
            break;
        }
        case Precision::I8: {
            int8_t data = static_cast<int8_t>(value);
            std::memcpy(baseOffset, &data, 1);
            break;
        }
        case Precision::BF16: {
            uint16_t data = ngraph::bfloat16(value).to_bits();
            std::memcpy(baseOffset, &data, 2);
            break;
        }
        case Precision::FP32: {
            std::memcpy(baseOffset, &value, sizeof(float));
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer does not support precision: " << prec;
            break;
        }
    }
}

// scale is float(outShape) / float(inShape)
// strictly consistent with onnx calc manner(div scale, not multiply inverse), given this is done offline
// the slight precison diff can produce obvious wrong value due to "nearest round" behavior for NN mode
inline float MKLDNNInterpolateNode::coordTransToInput(int outCoord, float scale, int inShape, int outShape) {
    if (scale == 1.0f || (inShape == outShape)) {
        return outCoord;
    }
    switch (coordTransMode) {
        case InterpolateCoordTransMode::half_pixel: {
            return (outCoord + 0.5f) / scale - 0.5f;
            break;
        }
        case InterpolateCoordTransMode::pytorch_half_pixel: {
            if (outShape > 1)
                return (outCoord + 0.5f) / scale - 0.5f;
            else
                return 0;
            break;
        }
        case InterpolateCoordTransMode::asymmetric: {
            return static_cast<float>(outCoord) / scale;
            break;
        }
        case InterpolateCoordTransMode::tf_half_pixel_for_nn: {
            return (outCoord + 0.5f) / scale;
            break;
        }
        case InterpolateCoordTransMode::align_corners: {
            if (outShape > 1)
                return outCoord * (static_cast<float>(inShape - 1) / static_cast<float>(outShape - 1));
            else
                return 0;
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' does not support specified coordinate transformation mode";
            break;
        }
    }
}

inline int MKLDNNInterpolateNode::nearestRound(float originCoord, bool isDownsample) {
    switch (nearestMode) {
        case InterpolateNearestMode::round_prefer_floor: {
            if (originCoord == (static_cast<int>(originCoord) + 0.5f))
                return static_cast<int>(std::floor(originCoord));
            else
                return static_cast<int>(std::round(originCoord));
            break;
        }
        case InterpolateNearestMode::round_prefer_ceil: {
            return static_cast<int>(std::round(originCoord));
            break;
        }
        case InterpolateNearestMode::floor: {
            return static_cast<int>(std::floor(originCoord));
            break;
        }
        case InterpolateNearestMode::ceil: {
            return static_cast<int>(std::ceil(originCoord));
            break;
        }
        case InterpolateNearestMode::simple: {
            if (isDownsample)
                return static_cast<int>(std::ceil(originCoord));
            else
                return static_cast<int>(originCoord);
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' does not support specified nearest round mode";
            break;
        }
    }
}

bool MKLDNNInterpolateNode::canFuse(const MKLDNNNodePtr& node) const {
    auto isOneOf = [&](EltwiseOpType alg, std::vector<EltwiseOpType> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    if (!mayiuse(cpu::sse42) || mode == InterpolateMode::linear) {
        return false;
    }

    if (node->getType() == Quantize) {
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get quantize node " << node->getName();
        return !quantizeNode->isBinarization();
    } else if (node->getType() == Eltwise) {
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode*>(node.get());
        if (eltwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get eltwise node " << node->getName();
        return isOneOf(eltwiseNode->getOpType(), {Prelu, Relu, Gelu, Elu, Logistic, BoundedRelu, Clamp,
                                                  Tanh, Swish, Hswish, Mish, Hsigmoid, Round, Linear, Abs, Square, Sqrt}) ||
                (eltwiseNode->getOpType() == MulAdd && eltwiseNode->getCnnLayer()->blobs.size() == 2);
    }

    return false;
}

bool MKLDNNInterpolateNode::created() const {
    return getType() == Interpolate;
}

REG_MKLDNN_PRIM_FOR(MKLDNNInterpolateNode, Interpolate);
