// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_interpolate_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_quantize_node.h"
#include "mkldnn_depthwise_node.h"
#include "mkldnn_activation_node.h"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>
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

        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        if (attr_.post_ops_.len_ != 0)
            mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);
        if (isa == cpu::avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        switch (jcp_.mode) {
            case InterpolateMode::nearest: {
                mov(reg_src, ptr[reg_params + GET_OFF(src)]);
                mov(reg_index, ptr[reg_params + GET_OFF(index)]);

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
                        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
                        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
                        mov(reg_src_aux, ptr[reg_params + GET_OFF(weight)]);

                        linear_onnx_planar();
                        break;
                    }
                    case InterpolateLayoutType::block:
                    case InterpolateLayoutType::by_channel: {
                        mov(reg_src, ptr[reg_params + GET_OFF(weight)]);
                        mov(reg_src_aux, ptr[reg_params + GET_OFF(weightR)]);
                        mov(reg_src_aux1, ptr[reg_params + GET_OFF(weightT)]);
                        mov(reg_src_aux2, ptr[reg_params + GET_OFF(weightB)]);
                        uni_vbroadcastss(vmm_weightL, ptr[reg_src]);
                        uni_vbroadcastss(vmm_weightR, ptr[reg_src_aux]);
                        uni_vbroadcastss(vmm_weightT, ptr[reg_src_aux1]);
                        uni_vbroadcastss(vmm_weightB, ptr[reg_src_aux2]);
                        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
                        mov(reg_src_aux, ptr[reg_params + GET_OFF(srcTR)]);
                        mov(reg_src_aux1, ptr[reg_params + GET_OFF(srcBL)]);
                        mov(reg_src_aux2, ptr[reg_params + GET_OFF(srcBR)]);

                        linear_onnx_c_gathered();
                        break;
                    }
                    default:
                        assert(!"unsupported memory layout for interpolate layer with linear_onnx mode.");
                }
                break;
            }
            case InterpolateMode::linear:
            case InterpolateMode::cubic: {
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
    Xbyak::Reg32 reg_index_oc = edx;

    Vmm vmm_val = Vmm(0);
    Xmm xmm_val = Xmm(0);
    Vmm vmm_index = Vmm(1);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_mask = Vmm(3);
    Vmm vmm_d_weights = Vmm(4);
    Vmm vmm_d_bias = Vmm(5);

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
            mov(reg_index_oc, dword[reg_index_h]);
            add(reg_src_h, reg_index_oc);  // reg_src_h now point to begin of row

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
                mov(reg_index_oc, dword[reg_index]);
                add(reg_src_aux, reg_index_oc);

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
            mov(reg_index_oc, dword[reg_index]);
            add(reg_src_aux, reg_index_oc);

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
            mov(reg_index_oc, dword[reg_index]);
            add(reg_src_aux, reg_index_oc);

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

            load_vector(vmm_valTL, ptr[reg_src], jcp_.src_dt);
            load_vector(vmm_valTR, ptr[reg_src_aux], jcp_.src_dt);
            load_vector(vmm_valBL, ptr[reg_src_aux1], jcp_.src_dt);
            load_vector(vmm_valBR, ptr[reg_src_aux2], jcp_.src_dt);

            // weightT * (srcTL * weightL + srcTR * weightR) +
            // weightB * (srcBL * weightL + srcBR * weightR);
            uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightR);
            uni_vmulps(vmm_valBR, vmm_valBR, vmm_weightR);
            uni_vfmadd231ps(vmm_valTR, vmm_valTL, vmm_weightL);
            uni_vfmadd231ps(vmm_valBR, vmm_valBL, vmm_weightL);
            uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightT);
            uni_vfmadd231ps(vmm_valTR, vmm_valBR, vmm_weightB);

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

                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightR);
                uni_vmulps(vmm_valBR, vmm_valBR, vmm_weightR);
                uni_vfmadd231ps(vmm_valTR, vmm_valTL, vmm_weightL);
                uni_vfmadd231ps(vmm_valBR, vmm_valBL, vmm_weightL);
                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightT);
                uni_vfmadd231ps(vmm_valTR, vmm_valBR, vmm_weightB);

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
                int src_stride = blk * jcp_.IW * jcp_.IH * jcp_.src_data_size;;
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

            uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightR);
            uni_vmulps(vmm_valBR, vmm_valBR, vmm_weightR);
            uni_vfmadd231ps(vmm_valTR, vmm_valTL, vmm_weightL);
            uni_vfmadd231ps(vmm_valBR, vmm_valBL, vmm_weightL);
            uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightT);
            uni_vfmadd231ps(vmm_valTR, vmm_valBR, vmm_weightB);

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

            // weightT * (srcTL * weightL + srcTR * weightR) +
            // weightB * (srcBL * weightL + srcBR * weightR);
            uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightR);
            uni_vmulps(vmm_valBR, vmm_valBR, vmm_weightR);
            uni_vfmadd231ps(vmm_valTR, vmm_valTL, vmm_weightL);
            uni_vfmadd231ps(vmm_valBR, vmm_valBL, vmm_weightL);
            uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightT);
            uni_vfmadd231ps(vmm_valTR, vmm_valBR, vmm_weightB);

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
            mov(reg_index_oc, dword[reg_index]);
            add(reg_src_aux1, reg_index_oc);
            load_scalar(xmm_valTL, ptr[reg_src_aux1], jcp_.src_dt);

            mov(reg_src_aux1, reg_src);
            mov(reg_index_oc, dword[reg_index + index_stride]);
            add(reg_src_aux1, reg_index_oc);
            load_scalar(xmm_valTR, ptr[reg_src_aux1], jcp_.src_dt);

            mov(reg_src_aux1, reg_src);
            mov(reg_index_oc, dword[reg_index + 2 * index_stride]);
            add(reg_src_aux1, reg_index_oc);
            load_scalar(xmm_valBL, ptr[reg_src_aux1], jcp_.src_dt);

            mov(reg_src_aux1, reg_src);
            mov(reg_index_oc, dword[reg_index + 3 * index_stride]);
            add(reg_src_aux1, reg_index_oc);
            load_scalar(xmm_valBR, ptr[reg_src_aux1], jcp_.src_dt);

            load_scalar(xmm_weightL, ptr[reg_src_aux], memory::f32);
            load_scalar(xmm_weightR, ptr[reg_src_aux + weight_stride], memory::f32);
            load_scalar(xmm_weightT, ptr[reg_src_aux + 2 * weight_stride], memory::f32);
            load_scalar(xmm_weightB, ptr[reg_src_aux + 3 * weight_stride], memory::f32);

            uni_vmulps(xmm_valTR, xmm_valTR, xmm_weightR);
            uni_vmulps(xmm_valBR, xmm_valBR, xmm_weightR);
            uni_vfmadd231ps(xmm_valTR, xmm_valTL, xmm_weightL);
            uni_vfmadd231ps(xmm_valBR, xmm_valBL, xmm_weightL);
            uni_vmulps(xmm_valTR, xmm_valTR, xmm_weightT);
            uni_vfmadd231ps(xmm_valTR, xmm_valBR, xmm_weightB);

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
                if (isa != cpu::sse42) {
                    vpmovzxwd(vmm_src, op);
                }
                uni_vpslld(vmm_src, vmm_src, 16);
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != memory::f32 && src_dt != data_type::bf16)
            uni_vcvtdq2ps(vmm_src, vmm_src);
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

    auto pushDesc = [&](memory::format dataFormat) {
        config.inConfs[DATA_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(DATA_ID)->getDims(), inputDataType, dataFormat);
        config.inConfs[TARGET_SHAPE_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(TARGET_SHAPE_ID)->getDims(), targetShapeType, memory::x);
        config.inConfs[SCALES_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(SCALES_ID)->getDims(), scalesType, memory::x);
        if (isAxesSpecified)
            config.inConfs[AXES_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(AXES_ID)->getDims(), axesType, memory::x);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, dataFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, dataFormat});
    };

    if (mode == InterpolateMode::nearest || mode == InterpolateMode::linear_onnx) {
        // blk and by_channel JIT kernel on sse42 or above machine
        if (mayiuse(cpu::sse42)) {
            if (getParentEdgeAt(DATA_ID)->getDims().ndims() == 4) {
                pushDesc(memory::nhwc);
                if (mayiuse(cpu::avx512_common)) {
                    pushDesc(memory::nChw16c);
                } else {
                    pushDesc(memory::nChw8c);
                }
            } else if (getParentEdgeAt(DATA_ID)->getDims().ndims() == 5 && mode == InterpolateMode::nearest) {
                pushDesc(memory::ndhwc);
                if (mayiuse(cpu::avx512_common)) {
                    pushDesc(memory::nCdhw16c);
                } else {
                    pushDesc(memory::nCdhw8c);
                }
            }
            if (fusedWith.empty()) {
                pushDesc(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(DATA_ID)->getDims()));
            }
        }

        // planar for 1.ref on machine without sse42(no fuse). 2.JIT kernel for f32 && avx2(gather).(with fuse)
        if (!mayiuse(cpu::sse42))
            pushDesc(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(DATA_ID)->getDims()));

        if (mayiuse(cpu::avx2) && inputPrec == Precision::FP32) {
            pushDesc(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(DATA_ID)->getDims()));
        }
    } else if (mode == InterpolateMode::linear || mode == InterpolateMode::cubic) {
        pushDesc(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(DATA_ID)->getDims()));
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
    jcp.IW = srcDim[dimSize - 1];
    jcp.IH = srcDim[dimSize - 2];

    if (MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims()) == selected_layout) {
        jcp.layout = InterpolateLayoutType::planar;
    } else if ((selected_layout == NHWC) || (selected_layout == NDHWC)) {
        jcp.layout = InterpolateLayoutType::by_channel;
    } else {
        jcp.layout = InterpolateLayoutType::block;
    }

    if (mode == InterpolateMode::nearest || mode == InterpolateMode::linear_onnx) {
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
            buidTblLinear(srcDimPad5d, dstDim5d, dataScales, LINEAR_KERNEL, antialias);
            break;
        }
        case InterpolateMode::cubic: {
            buidTblCubic(srcDimPad5d, dstDim5d, dataScales, cubeCoeff);
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
    float fz = (dimSize == 5) ? (1.f / dataScales[dimSize - 3]) : 1.f;
    float fy = 1.f / dataScales[dimSize - 2];
    float fx = 1.f / dataScales[dimSize - 1];
    size_t ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    indexTable.resize(OD + OH + OW);
    bool isDDownsample = (fz > 1) ? true : false;
    bool isHDownsample = (fy > 1) ? true : false;
    bool isWDownsample = (fx > 1) ? true : false;
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
    float fy = 1.f / dataScales[dimSize - 2];
    float fx = 1.f / dataScales[dimSize - 1];
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
        std::vector<int> index(scratchLen, 0);
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
void MKLDNNInterpolateNode::buidTblLinear(SizeVector& srcDimPad5d, SizeVector& dstDim5d,
                                            std::vector<float>& dataScales, int kernel_width, bool antialias) {
    int dimSize = srcDim.size();
    float fz = (dimSize == 5) ? (1.f / dataScales[dimSize - 3]) : 1.f;
    float fy = 1.f / dataScales[dimSize - 2];
    float fx = 1.f / dataScales[dimSize - 1];
    size_t ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    if (!(IW == OW && IH == OH && ID == OD)) {
        float ax = 1.0f / (antialias ? fx : 1.0f);
        float ay = 1.0f / (antialias ? fy : 1.0f);
        float az = 1.0f / (antialias ? fz : 1.0f);

        int rx = (fx < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
        int ry = (fy < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
        int rz = (fz < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

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
void MKLDNNInterpolateNode::buidTblCubic(SizeVector& srcDimPad5d, SizeVector& dstDim5d, std::vector<float>& dataScales, float cubicCoeff) {
    int dimSize = srcDim.size();
    float fy = 1.f / dataScales[dimSize - 2];
    float fx = 1.f / dataScales[dimSize - 1];
    int IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    int OH = dstDim5d[3], OW = dstDim5d[4];

    // idxNum for index, CUBIC_GRID_LEN for weight
    const int idxNum = 1;
    indexTable.resize((CUBIC_GRID_LEN + idxNum) * OW + (CUBIC_GRID_LEN + idxNum) * OH);
    int *xOrigin = static_cast<int*>(&indexTable[0]);
    float *xFactor = reinterpret_cast<float*>(&indexTable[OW]);
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

    int *yOrigin = static_cast<int*>(&indexTable[(CUBIC_GRID_LEN + idxNum) * OW]);
    float *yFactor = reinterpret_cast<float*>(&indexTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);
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

        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(node.get());
        if (depthwiseNode) {
            if (initWeights) {
                auto* depthwiseLayer = reinterpret_cast<WeightableLayer*>(depthwiseNode->getCnnLayer().get());
                MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(getChildEdgeAt(0)->getDims()[1], 16))});

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx]->Create(depthwiseDims, memory::data_type::f32, memory::format::x);

                PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::x,
                                                        depthwiseLayer->_weights->buffer(),
                                                        depthwiseLayer->_weights->size() *
                                                        MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                if (depthwiseNode->isBroadcast()) {
                    float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[0];
                    for (int i = 1; i < PostOpsIntBlobMemory[blob_idx]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                        static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[i] = broadcastValue;
                    }
                }

                if (depthwiseNode->getAlgorithm() == depthwise_scale_shift) {
                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    PostOpsIntBlobMemory[blob_idx + 1]->Create(depthwiseDims, memory::data_type::f32,
                                                               memory::format::x);
                    PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::x,
                                                                depthwiseLayer->_biases->buffer(),
                                                                depthwiseLayer->_biases->size() *
                                                                MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                    if (depthwiseNode->isBroadcast()) {
                        float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[0];
                        for (int i = 1; i < PostOpsIntBlobMemory[blob_idx + 1]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                            static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[i] = broadcastValue;
                        }
                    }

                    ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                    blob_idx += 2;
                }
            } else {
                ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                     nullptr,
                                     nullptr);
            }

            continue;
        }

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(), activationNode->getBeta());

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
        srcPadded = std::vector<uint8_t>(inShapePadBlock[0] * srcDataSize, 0);
        uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);

        parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], [&](int n, int c, int d, int h) {
            uint8_t *src = src_data_origin + (inShapeBlock[1] * n + inShapeBlock[2] * c + inShapeBlock[3] * d + inShapeBlock[4] * h) * srcDataSize;
            uint8_t *srcPad = src_data_pad + (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) +
                           inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4) * srcDataSize;
            cpu_memcpy(srcPad, src, srcDim5d[4] * srcDataSize);
        });
        src_data = src_data_pad;
    } else {
        src_data = src_data_origin;
    }

    size_t N = srcDimPad5d[0], C = srcDimPad5d[1], ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];
    std::vector<float> dataScales = getScales();
    if (dimSize > 2 && (dataScales[0] != 1.f || dataScales[1] != 1.f)) {
        THROW_IE_EXCEPTION << "Interpolate layer only supports resize on spatial dimensions(depth, height and width)";
    }
    float fz = (dimSize == 5) ? (1.f / dataScales[dimSize - 3]) : 1.f;
    float fy = 1.f / dataScales[dimSize - 2];
    float fx = 1.f / dataScales[dimSize - 1];
    Layout layout = getParentEdgeAt(DATA_ID)->getDesc().getLayout();
    bool isPlanar = (layout == NC || layout == NCHW || layout == NCDHW) ? true : false;

    switch (mode) {
        case InterpolateMode::nearest: {
            if (interpolateKernel) {
                if (isPlanar) {
                    NNPlanar(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else {
                    NNCGathered(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                }
            } else {
                NNRef(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
            }
            break;
        }
        case InterpolateMode::linear: {
            bool isDownsample = (fx > 1) || (fy > 1) || (fz > 1);
            int kernel_width = 2;
            linearInterpolation(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
            break;
        }
        case InterpolateMode::linear_onnx: {
            if (interpolateKernel) {
                if (isPlanar) {
                    linearOnnxPlanar(src_data, dst_data, N, C, IH, IW, fx, fy, OH, OW);
                } else {
                    linearOnnxCGathered(src_data, dst_data, N, C, IH, IW, fx, fy, OH, OW);
                }
            } else {
                linearOnnxRef(src_data, dst_data, N, C, IH, IW, fx, fy, OH, OW);
            }
            break;
        }
        case InterpolateMode::cubic: {
            cubic(src_data, dst_data, N, C, IH, IW, fx, fy, OH, OW, cubeCoeff);
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer has unsupported interpolate mode: " << mode;
        }
    }
}

// for ndhwc and nCdhw8c[16c]
// input may be f32/bf16/int8, fused->output varies
void MKLDNNInterpolateNode::NNCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW) {
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
                arg.src = in_ptr_dh;
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
                    arg.src = in_ptr_cbd + blk_size * IW * index_h[h] * srcDataSize;
                    arg.index = static_cast<int*>(&(index_w_kernel[0]));
                    arg.work_amount = static_cast<size_t>(OW);
                    arg.oc_off = cb * blk_size;
                    (*interpolateKernel)(&arg);
                }
            });
        }
    }  // batch end
}

void MKLDNNInterpolateNode::NNPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW) {
    int *index_d = static_cast<int*>(&indexTable[0]);
    int *index_h = static_cast<int*>(&indexTable[OD]);
    int *index_w = static_cast<int*>(&indexTable[OD + OH]);

    // index_h * IW * srcDataSize
    for (int oh = 0; oh < OH; oh++) {
        index_h[oh] *= IW;
        index_h[oh] *= srcDataSize;
    }
    // index_w * srcDataSize
    for (int ow = 0; ow < OW; ow++) {
        index_w[ow] *= srcDataSize;
    }

    parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        const uint8_t *in_ptr = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]) * srcDataSize;
        uint8_t *out_ptr = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od) * dstDataSize;

        auto arg = jit_interpolate_call_args();
        arg.src = in_ptr;
        arg.dst = out_ptr;
        arg.index = index_h;  // need index_h and index_w in kernel, it's in continous memory so one param
        arg.oc_off = static_cast<size_t>(c);
        // work_amount is OH(out loop) and OW(inner loop), can get in kernel from jcp.
        (*interpolateKernel)(&arg);
    });
}

void MKLDNNInterpolateNode::NNRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW) {
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

void MKLDNNInterpolateNode::linearOnnxPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW,
                                          float fx, float fy, int OH, int OW) {
    int *index = static_cast<int*>(&indexTable[0]);
    int eltInGrid = 4;
    int scratchLen = rnd_up(eltInGrid * OW * OH, 16);
    float *weight = reinterpret_cast<float*>(&indexTable[scratchLen]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        uint8_t *out_ptr_nc = out_ptr_ + (OH * OW * C * b + OH * OW * c) * dstDataSize;
        const uint8_t *in_ptr_nc = in_ptr_ + (IH * IW * C * b + IH * IW * c) * srcDataSize;
        auto arg = jit_interpolate_call_args();
        arg.src = in_ptr_nc;
        arg.index = static_cast<int*>(&index[0]);
        arg.weight = static_cast<float*>(&weight[0]);
        arg.dst = out_ptr_nc;
        arg.work_amount = OW * OH;
        arg.oc_off = c;
        (*interpolateKernel)(&arg);
    });
}

void MKLDNNInterpolateNode::linearOnnxCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW,
                                          float fx, float fy, int OH, int OW) {
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

    if (isByChannel) {
        parallel_for3d(B, OH, OW, [&](size_t b, size_t h, size_t w) {
            uint8_t *out_ptr_nhw = out_ptr_ + (OH * OW * C * b + OW * C * h + C * w) * dstDataSize;
            const uint8_t *in_ptr_n = in_ptr_ + (IH * IW * C * b) * srcDataSize;
            auto arg = jit_interpolate_call_args();
            arg.src = in_ptr_n + (indexTop[h] * IW * C + indexLeft[w] * C) * srcDataSize;
            arg.srcTR = in_ptr_n + (indexTop[h] * IW * C + indexRight[w] * C) * srcDataSize;
            arg.srcBL = in_ptr_n + (indexBottom[h] * IW * C + indexLeft[w] * C) * srcDataSize;
            arg.srcBR = in_ptr_n + (indexBottom[h] * IW * C + indexRight[w] * C) * srcDataSize;
            arg.weight = static_cast<float*>(&weightLeft[w]);
            arg.weightR = static_cast<float*>(&weightRight[w]);
            arg.weightT = static_cast<float*>(&weightTop[h]);
            arg.weightB = static_cast<float*>(&weightBottom[h]);
            arg.dst = out_ptr_nhw;
            arg.work_amount = C;
            arg.oc_off = 0;
            (*interpolateKernel)(&arg);
        });
    } else {
        size_t blkSize = mayiuse(cpu::avx512_common) ? 16 : 8;
        size_t CB = div_up(C, blkSize);
        parallel_for3d(B, OH, OW, [&](size_t b, size_t h, size_t w) {
            uint8_t *out_ptr_nhw = out_ptr_ + (CB * OH * OW * blkSize * b + OW * blkSize * h + blkSize * w) * dstDataSize;
            const uint8_t *in_ptr_n = in_ptr_ + (CB * IH * IW * blkSize * b) * srcDataSize;
            auto arg = jit_interpolate_call_args();
            arg.src = in_ptr_n + (indexTop[h] * IW * blkSize + indexLeft[w] * blkSize) * srcDataSize;
            arg.srcTR = in_ptr_n + (indexTop[h] * IW * blkSize + indexRight[w] * blkSize) * srcDataSize;
            arg.srcBL = in_ptr_n + (indexBottom[h] * IW * blkSize + indexLeft[w] * blkSize) * srcDataSize;
            arg.srcBR = in_ptr_n + (indexBottom[h] * IW * blkSize + indexRight[w] * blkSize) * srcDataSize;
            arg.weight = static_cast<float*>(&weightLeft[w]);
            arg.weightR = static_cast<float*>(&weightRight[w]);
            arg.weightT = static_cast<float*>(&weightTop[h]);
            arg.weightB = static_cast<float*>(&weightBottom[h]);
            arg.dst = out_ptr_nhw;
            arg.work_amount = CB;
            arg.oc_off = 0;
            (*interpolateKernel)(&arg);
        });
    }
}

void MKLDNNInterpolateNode::linearOnnxRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW,
                                          float fx, float fy, int OH, int OW) {
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

    float ax = 1.0f / (antialias ? fx : 1.0f);
    float ay = 1.0f / (antialias ? fy : 1.0f);
    float az = 1.0f / (antialias ? fz : 1.0f);

    int rx = (fx < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
    int ry = (fy < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
    int rz = (fz < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

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


void MKLDNNInterpolateNode::cubic(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW,
                                    float fx, float fy, int OH, int OW, float a) {
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

// scale is float(inShape) / float(outShape)
// nearest mode need to be strictly consistent with onnx calc manner(div scale, not multiply inverse),
// the slight precison diff can produce obvious wrong value due to "nearest round" behavior for NN mode
inline float MKLDNNInterpolateNode::coordTransToInput(int outCoord, float scale, int inShape, int outShape) {
    if ((scale == 1.f) || (inShape == outShape)) {
        return static_cast<float>(outCoord);
    }
    if (mode == InterpolateMode::nearest) {
        scale = 1.f / scale;
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
                    return outCoord * static_cast<float>(inShape - 1) / static_cast<float>(outShape - 1);
                else
                    return 0;
                break;
            }
            default: {
                THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' does not support specified coordinate transformation mode";
                break;
            }
        }
    } else {
        switch (coordTransMode) {
            case InterpolateCoordTransMode::half_pixel: {
                return (outCoord + 0.5f) * scale - 0.5f;
                break;
            }
            case InterpolateCoordTransMode::pytorch_half_pixel: {
                if (outShape > 1)
                    return (outCoord + 0.5f) * scale - 0.5f;
                else
                    return 0;
                break;
            }
            case InterpolateCoordTransMode::asymmetric: {
                return outCoord * scale;
                break;
            }
            case InterpolateCoordTransMode::tf_half_pixel_for_nn: {
                return (outCoord + 0.5f) * scale;
                break;
            }
            case InterpolateCoordTransMode::align_corners: {
                if (outShape > 1)
                    return outCoord * static_cast<float>(inShape - 1) / static_cast<float>(outShape - 1);
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
    auto isOneOf = [](mkldnn::algorithm alg, std::vector<mkldnn::algorithm> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    if (!mayiuse(cpu::sse42))
        return false;
    if (mode == InterpolateMode::linear || mode == InterpolateMode::cubic)
        return false;

    if (node->getType() == Quantize) {
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();
        return !quantizeNode->isBinarization();
    } else if (node->getType() == Depthwise) {
        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode*>(node.get());
        if (depthwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get depthwise layer " << node->getName();
        return ((depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_scale_shift && depthwiseNode->isWithBiases()) ||
                (depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_prelu));
    } else if (node->getType() == Activation) {
        auto* activationNode = dynamic_cast<MKLDNNActivationNode*>(node.get());
        if (activationNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get activation layer " << node->getName();
        return isOneOf(activationNode->getAlgorithm(), {eltwise_relu, eltwise_gelu, eltwise_elu, eltwise_logistic,
            eltwise_bounded_relu, eltwise_clamp, eltwise_tanh, eltwise_swish, eltwise_hswish, eltwise_mish, eltwise_linear,
            eltwise_abs, eltwise_square, eltwise_sqrt});
    }
    return false;
}

bool MKLDNNInterpolateNode::created() const {
    return getType() == Interpolate;
}

REG_MKLDNN_PRIM_FOR(MKLDNNInterpolateNode, Interpolate);
