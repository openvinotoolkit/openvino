// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_mvn_node.h"

#include <algorithm>
#include <string>
#include <vector>

#include "mkldnn_fake_quantize_node.h"
#include "mkldnn_eltwise_node.h"
#include <mkldnn_extension_utils.h>
#include "utils/bfloat16.hpp"
#include "ie_parallel.hpp"
#include "emitters/jit_load_store_emitters.hpp"
#include "emitters/jit_bf16_emitters.hpp"

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/jit_uni_eltwise.hpp>
#include <cpu/x64/jit_uni_depthwise_injector.hpp>
#include <cpu/x64/jit_uni_quantization_injector.hpp>
#include <cpu/x64/jit_uni_eltwise_injector.hpp>

#include <ngraph/opsets/opset6.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_mvn_call_args, field)

// some utility functions
static inline bool isFloatCompatible(Precision prc) {
    return Precision::FP32 == prc || Precision::BF16 == prc;
}

static inline bool isFloatCompatible(memory::data_type type) {
    return memory::data_type::f32 == type || memory::data_type::bf16 == type;
}

// normalize_variance = false : src->mean
// normalize_variance = true : src+mean->variance:sqr(x-mean)
template <cpu_isa_t isa>
struct jit_uni_mvn_mean_variance_kernel_f32 : public jit_uni_mvn_mean_variance_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_mvn_mean_kernel_f32)

    explicit jit_uni_mvn_mean_variance_kernel_f32(jit_mvn_config_params jcp) : jit_uni_mvn_mean_variance_kernel(jcp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        load_emitter.reset(new jit_load_emitter(this, isa, nullptr));

        this->preamble();
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        if (jcp_.normalize_variance) {
            mov(reg_mean, ptr[reg_params + GET_OFF(mean)]);
            mov(reg_variance, ptr[reg_params + GET_OFF(variance)]);
            uni_vpxor(vmm_variance, vmm_variance, vmm_variance);
        } else {
            mov(reg_sum, ptr[reg_params + GET_OFF(sum)]);
            uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
        }
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_stride, ptr[reg_params + GET_OFF(src_stride)]);
        mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

        if (jcp_.normalize_variance) {
            if (jcp_.planar_layout || jcp_.across_channels) {
                uni_vbroadcastss(vmm_mean, ptr[reg_mean]);
            } else {
                uni_vmovups(vmm_mean, ptr[reg_mean]);
            }
        }

        tail_num = jcp_.planar_layout ? (jcp_.D * jcp_.H * jcp_.W) - ((jcp_.D * jcp_.H * jcp_.W) / step) * step :
                                        jcp_.C - (jcp_.C / step) * step;

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};

        if (jcp_.planar_layout) {
            worker_unroll();
            if (tail_num != 0) {
                worker_tail_planar();
            }

            // hsum+store
            if (!jcp_.normalize_variance && !isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_sum, vmm_sum);
            Vmm vmm_dst = jcp_.normalize_variance ? vmm_variance : vmm_sum;
            if (isa == cpu::x64::sse41) {
                hsum_store(vmm_dst);
            } else if (isa == cpu::x64::avx2) {
                Xbyak::Ymm ymm_sum = Xbyak::Ymm(vmm_dst.getIdx());
                vextractf128(xmm_aux1, ymm_sum, 0);
                vextractf128(xmm_aux2, ymm_sum, 1);
                addps(xmm_aux1, xmm_aux2);
                hsum_store(xmm_aux1);
            } else {
                Xbyak::Zmm zmm_sum = Xbyak::Zmm(vmm_dst.getIdx());
                vextractf32x4(xmm_aux1, zmm_sum, 0);
                vextractf32x4(xmm_aux2, zmm_sum, 1);
                addps(xmm_aux1, xmm_aux2);
                vextractf32x4(xmm_aux2, zmm_sum, 2);
                vextractf32x4(xmm_aux3, zmm_sum, 3);
                addps(xmm_aux2, xmm_aux3);
                addps(xmm_aux1, xmm_aux2);
                hsum_store(xmm_aux1);
            }
        } else {
            // blk+nspc
            int repeats = (isa == cpu::x64::sse41) ? 2 : 1; // block size is also 8 on cpu::x64::sse41 with two step process
            int sse42_step = 4;
            for (int i = 0; i < repeats; i++) {
                int offset_sse42 = i * sse42_step;
                if (i > 0) {
                    mov(reg_src, ptr[reg_params + GET_OFF(src)]);
                    mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

                    add(reg_src, offset_sse42 * jcp_.src_data_size);

                    if (jcp_.normalize_variance) {
                        // mean and vaiance for variance kernel
                        if (!jcp_.across_channels) {
                            // mean is bc when across_channel, no need shift
                            add(reg_mean, offset_sse42 * sizeof(float));
                            uni_vmovups(vmm_mean, ptr[reg_mean]);
                        }
                        add(reg_variance, offset_sse42 * sizeof(float));
                        uni_vpxor(vmm_variance, vmm_variance, vmm_variance);
                    } else {
                        // sum for mean kernel
                        add(reg_sum, offset_sse42 * sizeof(float));
                        uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
                    }
                    add(reg_oc_off, offset_sse42 * sizeof(float));
                }

                Xbyak::Label label_empty_2half_sse42;
                if (tail_num == 0) {
                    cmp(reg_oc_off, static_cast<int>(jcp_.C * sizeof(float)));
                    jae(label_empty_2half_sse42, T_NEAR);

                    worker_unroll();
                } else {
                    // maybe tail blk
                    cmp(reg_oc_off, static_cast<int>(jcp_.C * sizeof(float)));
                    jae(label_empty_2half_sse42, T_NEAR);

                    Xbyak::Label label_full_size;
                    Xbyak::Label label_size_end;
                    cmp(reg_oc_off, static_cast<int>((jcp_.C - step) * sizeof(float)));
                    jle(label_full_size, T_NEAR);

                    // no need care and fill rest
                    // for per_channel, do not use tail mean(variance), do not store computed tail values.
                    // for across_channel, partial sum for tail one time out of kernel from perf.
                    worker_unroll(true);

                    jmp(label_size_end, T_NEAR);
                    L(label_full_size);
                    {
                        worker_unroll();
                    }
                    L(label_size_end);
                }

                // add input_base value and store for per_channel
                // store for across_channels
                if (jcp_.normalize_variance) {
                    if (!jcp_.across_channels) {
                        uni_vmovups(vmm_val, ptr[reg_variance]);
                        uni_vaddps(vmm_variance, vmm_variance, vmm_val);
                    }
                    uni_vmovups(ptr[reg_variance], vmm_variance);
                } else {
                    if (!isFloatCompatible(jcp_.src_prc))  // add with int for int-family data type, other compute go with float
                        uni_vcvtdq2ps(vmm_sum, vmm_sum);

                    if (!jcp_.across_channels) {
                        uni_vmovups(vmm_val, ptr[reg_sum]);
                        uni_vaddps(vmm_sum, vmm_sum, vmm_val);
                    }
                    uni_vmovups(ptr[reg_sum], vmm_sum);
                }

                L(label_empty_2half_sse42);
            }
        }

        this->postamble();

        load_emitter->emit_data();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int step = vlen / sizeof(float);
    int tail_num = 0;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_mean = r9;
    Xbyak::Reg64 reg_variance = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_stride = r12;
    Xbyak::Reg64 reg_sum = reg_mean;
    Xbyak::Reg64 reg_params = abi_param1;
    Xbyak::Reg64 reg_load_table = r13;
    Xbyak::Reg64 reg_load_store_mask = r14;
    Xbyak::Reg64 reg_aux = r15;

    Xbyak::Reg64 reg_oc_off = rax;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_mean = Vmm(1);
    Vmm vmm_variance = Vmm(2);
    Vmm vmm_sum = vmm_mean;
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(3);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(4);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(5);
    Vmm vmm_zero = Vmm(6);

    Xbyak::Opmask k_mask = Xbyak::Opmask(7);

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;

    std::vector<size_t> load_pool_gpr_idxs;

    inline void worker_full_size() {
        Precision dst_prc = isFloatCompatible(jcp_.src_prc) ? Precision::FP32 : Precision::I32;
        load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                            std::make_shared<load_emitter_context>(jcp_.src_prc, dst_prc, step),
                            {}, {load_pool_gpr_idxs});

        if (jcp_.normalize_variance) {
            // all with float
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_val, vmm_val);

            uni_vsubps(vmm_val, vmm_val, vmm_mean);
            uni_vfmadd231ps(vmm_variance, vmm_val, vmm_val);
        } else {
            // for sum, int execute prc for int-family data type
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vpaddd(vmm_sum, vmm_sum, vmm_val);
            else
                uni_vaddps(vmm_sum, vmm_sum, vmm_val);
        }
    }

    inline void worker_tail_blk() {
        Precision dst_prc = isFloatCompatible(jcp_.src_prc) ? Precision::FP32 : Precision::I32;
        load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                            std::make_shared<load_emitter_context>(jcp_.src_prc, dst_prc, tail_num),
                            {}, {load_pool_gpr_idxs});

        if (jcp_.normalize_variance) {
            // all with float
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_val, vmm_val);

            uni_vsubps(vmm_val, vmm_val, vmm_mean);
            uni_vfmadd231ps(vmm_variance, vmm_val, vmm_val);
        } else {
            // for sum, int execute prc for int-family data type
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vpaddd(vmm_sum, vmm_sum, vmm_val);
            else
                uni_vaddps(vmm_sum, vmm_sum, vmm_val);
        }
    }

    inline void worker_unroll(bool is_tail = false) {
        Xbyak::Label loop_label;
        Xbyak::Label loop_end_label;
        L(loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(loop_end_label, T_NEAR);

            if (!jcp_.planar_layout && is_tail) {
                worker_tail_blk();
            } else {
                worker_full_size();
            }

            add(reg_src, reg_stride);
            sub(reg_work_amount, 1);

            jmp(loop_label, T_NEAR);
        }
        L(loop_end_label);
    }

    inline void worker_tail_planar() {
        Precision dst_prc = isFloatCompatible(jcp_.src_prc) ? Precision::FP32 : Precision::I32;
        load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                std::make_shared<load_emitter_context>(jcp_.src_prc, dst_prc, tail_num, 0, true),
                                {}, {load_pool_gpr_idxs});

        if (jcp_.normalize_variance) {
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_val, vmm_val);

            uni_vsubps(vmm_val, vmm_val, vmm_mean);

            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
            if (isa == cpu::x64::sse41) {
                uint8 imm = 1;
                imm = ~((imm << tail_num) - imm);
                blendps(vmm_val, vmm_zero, imm);
            } else if (isa == cpu::x64::avx2) {
                uint8 imm = 1;
                imm = ~((imm << tail_num) - imm);
                vblendps(vmm_val, vmm_val, vmm_zero, imm);
            } else if (isa == cpu::x64::avx512_common) {
                uint64_t tail_mask = 1;
                tail_mask = ~((tail_mask << tail_num) - tail_mask);
                mov(reg_aux, tail_mask);
                kmovq(k_mask, reg_aux);
                vblendmps(vmm_val | k_mask, vmm_val, vmm_zero);
            }

            uni_vfmadd231ps(vmm_variance, vmm_val, vmm_val);
        } else {
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vpaddd(vmm_sum, vmm_sum, vmm_val);
            else
                uni_vaddps(vmm_sum, vmm_sum, vmm_val);
        }
    }

    inline void hsum_store(Xbyak::Xmm xmm_sum) {
        movshdup(xmm_aux3, xmm_sum);  //  sum:1,2,3,4; aux3:2,2,4,4
        addps(xmm_sum, xmm_aux3);     //  sum:1+2,2+2,3+4,4+4
        movhlps(xmm_aux3, xmm_sum);   //  aux3:3+4,4+4,4,4
        addps(xmm_sum, xmm_aux3);     //  sum:1+2+3+4,...
        if (jcp_.normalize_variance) {
            movss(ptr[reg_variance], xmm_sum);
        } else {
            movss(ptr[reg_sum], xmm_sum);
        }
    }
};

// mean,variance->mvn
template <cpu_isa_t isa>
struct jit_uni_mvn_kernel_f32 : public jit_uni_mvn_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_mvn_kernel_f32)

    explicit jit_uni_mvn_kernel_f32(jit_mvn_config_params jcp, const mkldnn_primitive_attr &attr) : jit_uni_mvn_kernel(jcp, attr), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta, post_op.eltwise.scale));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this, post_op.depthwise.alg));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        load_emitter.reset(new jit_load_emitter(this, isa, nullptr));
        store_emitter.reset(new jit_store_emitter(this, isa, nullptr));

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_mean, ptr[reg_params + GET_OFF(mean)]);
        if (jcp_.normalize_variance)
            mov(reg_variance_inv, ptr[reg_params + GET_OFF(variance)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_src_stride, ptr[reg_params + GET_OFF(src_stride)]);
        mov(reg_dst_stride, ptr[reg_params + GET_OFF(dst_stride)]);
        mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

        if (jcp_.planar_layout || jcp_.across_channels) {
            uni_vbroadcastss(vmm_mean, ptr[reg_mean]);
            if (jcp_.normalize_variance)
                uni_vbroadcastss(vmm_variance_inv, ptr[reg_variance_inv]);
        } else {
            uni_vmovups(vmm_mean, ptr[reg_mean]);
            if (jcp_.normalize_variance)
                uni_vmovups(vmm_variance_inv, ptr[reg_variance_inv]);
        }

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        tail_num = jcp_.planar_layout ? (jcp_.D * jcp_.H * jcp_.W) - ((jcp_.D * jcp_.H * jcp_.W) / step) * step :
                                        jcp_.C - (jcp_.C / step) * step;

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        if (jcp_.planar_layout) {
            worker_mvn_unroll();
            if (tail_num != 0) {
                worker_mvn(true);
            }
        } else {
            // blk+nspc
            int repeats = (isa == cpu::x64::sse41) ? 2 : 1;  // block size is also 8 on cpu::x64::sse41
            for (int i = 0; i < repeats; i++) {
                int offset_sse42 = i * 4;
                if (i > 0) {
                    // reset modified input
                    mov(reg_src, ptr[reg_params + GET_OFF(src)]);
                    mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
                    mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

                    add(reg_src, offset_sse42 * jcp_.src_data_size);
                    add(reg_dst, offset_sse42 * jcp_.dst_data_size);
                    add(reg_oc_off, offset_sse42 * sizeof(float));

                    if (!jcp_.across_channels) {
                        add(reg_mean, offset_sse42 * sizeof(float));
                        uni_vmovups(vmm_mean, ptr[reg_mean]);
                        if (jcp_.normalize_variance) {
                            add(reg_variance_inv, offset_sse42 * sizeof(float));
                            uni_vmovups(vmm_variance_inv, ptr[reg_variance_inv]);
                        }
                    }
                }

                Xbyak::Label label_empty_2half_sse42;
                if (tail_num == 0) {
                    cmp(reg_oc_off, static_cast<int>(jcp_.C * sizeof(float)));
                    jae(label_empty_2half_sse42, T_NEAR);
                    worker_mvn_unroll();
                } else {
                    cmp(reg_oc_off, static_cast<int>(jcp_.C * sizeof(float)));
                    jae(label_empty_2half_sse42, T_NEAR);

                    Xbyak::Label label_full_size_block;
                    Xbyak::Label label_size_end;

                    cmp(reg_oc_off, static_cast<int>((jcp_.C - step) * sizeof(float)));
                    jle(label_full_size_block, T_NEAR);

                    worker_mvn_unroll(true);
                    jmp(label_size_end, T_NEAR);

                    L(label_full_size_block);
                    {
                        worker_mvn_unroll();
                    }
                    L(label_size_end);
                }
                L(label_empty_2half_sse42);
            }
        }

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int step = vlen / sizeof(float);
    int tail_num = 0;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_mean = r9;
    Xbyak::Reg64 reg_variance_inv = r10;
    Xbyak::Reg64 reg_dst = r11;
    Xbyak::Reg64 reg_work_amount = r12;
    Xbyak::Reg64 reg_src_stride = r13;
    Xbyak::Reg64 reg_dst_stride = r14;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = rdx;

    Xbyak::Reg64 reg_load_table = r15;
    Xbyak::Reg64 reg_load_store_mask = rbp;

    Vmm vmm_val = Vmm(1);
    Vmm vmm_mean = Vmm(0);
    Vmm vmm_variance_inv = Vmm(2);
    Vmm vmm_zero = Vmm(3);

    Vmm vmm_d_weights = Vmm(5);
    Vmm vmm_d_bias = Vmm(6);

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;
    std::vector<size_t> load_pool_gpr_idxs;

    inline void worker_mvn(bool is_tail) {
        int elt_num = is_tail ? tail_num : step;
        load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
            std::make_shared<load_emitter_context>(jcp_.src_prc, Precision::FP32, elt_num),
            {}, {load_pool_gpr_idxs});

        uni_vsubps(vmm_val, vmm_val, vmm_mean);
        if (jcp_.normalize_variance)
            uni_vmulps(vmm_val, vmm_val, vmm_variance_inv);

        apply_post_ops(jcp_.dst_prc, jcp_.planar_layout);

        store_emitter->emit_code({static_cast<size_t>(vmm_val.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
            std::make_shared<store_emitter_context>(Precision::FP32, jcp_.dst_prc, elt_num),
            {store_pool_vec_idxs}, {store_pool_gpr_idxs});
    }

    inline void worker_mvn_unroll(bool is_tail = false) {
        Xbyak::Label mvn_loop_label;
        Xbyak::Label mvn_loop_end_label;

        L(mvn_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(mvn_loop_end_label, T_NEAR);

            worker_mvn(is_tail);

            add(reg_src, reg_src_stride);
            add(reg_dst, reg_dst_stride);
            sub(reg_work_amount, 1);

            jmp(mvn_loop_label, T_NEAR);
        }
        L(mvn_loop_end_label);
    }

    void apply_post_ops(InferenceEngine::Precision dst_prc, bool is_broadcast) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));
                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1, reg_d_weights, reg_d_bias, is_broadcast);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || isFloatCompatible(dst_prc) || i != p.len() - 1;
                int s_idx = vmm_val.getIdx();

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_inj_idx++;
            }
        }
    }
};
//////////////////////////////////////////////////////////////////////////////////

bool MKLDNNMVNNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_output_partial_shape(0).rank().is_dynamic()) {
            errorMessage = "Unsupported dynamic input rank.";
            return false;
        }
        const auto& inDataRank = op->get_output_partial_shape(0).rank().get_length();
        if (inDataRank < 1 || inDataRank > 5) {
            errorMessage = "First input accepts ranks from 1 to 5. Actual: " + std::to_string(inDataRank);
            return false;
        }

        if (auto mvnOp = ngraph::as_type_ptr<const ngraph::op::v6::MVN>(op)) {
            auto axesOp = ngraph::as_type_ptr<ngraph::op::Constant>(mvnOp->get_input_node_shared_ptr(1));
            if (!axesOp) {
                errorMessage = "Constant expected as the second input.";
                return false;
            }

            auto epsMode = mvnOp->get_eps_mode();
            if (epsMode != ngraph::op::MVNEpsMode::INSIDE_SQRT &&
                    epsMode != ngraph::op::MVNEpsMode::OUTSIDE_SQRT) {
                errorMessage = std::string("Just INSIDE_SQRT and OUTSIDE_SQRT epsilon mods are supported. Actual: ") +
                        std::to_string(static_cast<int>(epsMode));
                return false;
            }
            // Validates MVN node axes to check whether it can be executed on the current CPU implementation.
            // Supported cases:
            // 1D: axes: [0]
            // 2D: axes: [1]
            // 3D: axes: [1,2], [2]
            // 4D: axes: [1,2,3], [2,3]
            // 5D: axes: [1,2,3,4], [2,3,4]
            auto axesVal = axesOp->cast_vector<int>();
            for (int& axe : axesVal)
                axe = axe < 0 ? axe + inDataRank : axe;
            std::sort(axesVal.begin(), axesVal.end());
            if (inDataRank == 1) {
                if (axesVal.size() != 1 || axesVal[0] != 0) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
            } else {
                if (inDataRank > 5 || (inDataRank != axesVal.size() + 1 && inDataRank != axesVal.size() + 2)) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
                int value = inDataRank - 1;
                for (int i = axesVal.size() - 1; i >= 0; i--, value--) {
                    if (axesVal[i] != value) {
                        errorMessage = "Unsupported axes.";
                        return false;
                    }
                }
            }
        } else if (auto mvnOp = ngraph::as_type_ptr<const ngraph::op::v0::MVN>(op)) {
        } else {
            errorMessage = "Node is not an instance of the MVN operation.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMVNNode::MKLDNNMVNNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const ngraph::Shape& inDataShape = op->input_value(0).get_shape();
    if (auto mvnOp = ngraph::as_type_ptr<ngraph::op::v6::MVN>(op)) {
        normalizeVariance_ = mvnOp->get_normalize_variance();
        epsValue_ = mvnOp->get_eps();
        epsMode_ = INSIDE_SQRT;
        if (mvnOp->get_eps_mode() == ngraph::op::MVNEpsMode::OUTSIDE_SQRT) {
            epsMode_ = OUTSIDE_SQRT;
        }

        acrossChannels_ = false;
        const auto& inDataShapeSize = inDataShape.size();
        if (inDataShapeSize == mvnOp->input_value(1).get_shape()[0] + 1 || inDataShapeSize == 1)
            acrossChannels_ = true;
    } else if (auto mvnOp = ngraph::as_type_ptr<ngraph::op::v0::MVN>(op)) {
        normalizeVariance_ = mvnOp->get_normalize_variance();
        epsValue_ = mvnOp->get_eps();
        epsMode_ = INSIDE_SQRT;
        acrossChannels_ = mvnOp->get_across_channels();
    }
}

void MKLDNNMVNNode::getSupportedDescriptors() {
}

void MKLDNNMVNNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    setPostOps(attr, true);

    Precision inputPrecision = getOriginalInputPrecisionAtPort(0);
    Precision outputPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!mayiuse(avx512_core)) {
        if (outputPrecision == Precision::BF16)
            outputPrecision = Precision::FP32;
    }

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    // ref with float planar and no fusion
    if (!mayiuse(cpu::x64::sse41)) {
        inputPrecision = outputPrecision = Precision::FP32;
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    input_prec = inputPrecision;
    output_prec = outputPrecision;
    src_data_size = MKLDNNExtensionUtils::sizeOfDataType(inputDataType);
    dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(outputDataType);

    bool canBeInplace = (src_data_size == dst_data_size) &&
                        (getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1) &&
                        !getParentEdgeAt(0)->getParent()->isConstant();

    const size_t inputsNum = getParentEdges().size();
    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(inputsNum);
    config.outConfs.resize(1);
    config.inConfs[0].constant = false;
    config.outConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.outConfs[0].inPlace = canBeInplace ? 0 : -1;
    if (inputsNum == 2) {
        config.inConfs[1].desc = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(1)->getShape().getStaticMklDims(), memory::data_type::s32,
                                                               MKLDNNMemory::GetPlainFormatByRank(getParentEdgeAt(1)->getShape().getRank()));
        config.inConfs[1].constant = true;
    }

    auto pushDesc = [&](memory::format_tag format, impl_desc_type impl_type) {
        config.inConfs[0].desc = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(0)->getShape().getStaticMklDims(), inputDataType, format);
        config.outConfs[0].desc = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(0)->getShape().getStaticMklDims(), outputDataType, format);
        supportedPrimitiveDescriptors.push_back({config, impl_type});
    };

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    if (mayiuse(cpu::x64::sse41)) {
        // nspc
        if (getParentEdgeAt(0)->getShape().getRank() == 4) {
            pushDesc(memory::format_tag::nhwc, impl_type);
        } else if (getParentEdgeAt(0)->getShape().getRank() == 5) {
            pushDesc(memory::format_tag::ndhwc, impl_type);
        }
        // blk
        if (impl_desc_type::jit_avx512 == impl_type) {
            if (getParentEdgeAt(0)->getShape().getRank() == 4) {
                pushDesc(memory::format_tag::nChw16c, impl_type);
            } else if (getParentEdgeAt(0)->getShape().getRank() == 5) {
                pushDesc(memory::format_tag::nCdhw16c, impl_type);
            }
        } else if (impl_desc_type::jit_avx2 ==  impl_type || impl_desc_type::jit_sse42 == impl_type) {
            if (getParentEdgeAt(0)->getShape().getRank() == 4) {
                pushDesc(memory::format_tag::nChw8c, impl_type);
            } else if (getParentEdgeAt(0)->getShape().getRank() == 5) {
                pushDesc(memory::format_tag::nCdhw8c, impl_type);
            }
        }
    }

    // planar
    if (canBeInplace)
        config.inConfs[0].inPlace = 0;
    pushDesc(MKLDNNMemory::GetPlainFormatByRank(getChildEdgeAt(0)->getShape().getRank()), impl_type);
}

void MKLDNNMVNNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";

    const SizeVector in_dims = getParentEdgeAt(0)->getShape().getStaticDims();
    transformTo5DCase(in_dims);
    auto selectedPD = getSelectedPrimitiveDescriptor();
    auto jcp = jit_mvn_config_params();
    jcp.src_prc = selectedPD->getConfig().inConfs[0].desc->getPrecision();
    jcp.dst_prc = selectedPD->getConfig().outConfs[0].desc->getPrecision();
    jcp.src_data_size = MKLDNNExtensionUtils::sizeOfDataType(MKLDNNExtensionUtils::IEPrecisionToDataType(jcp.src_prc));
    jcp.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(MKLDNNExtensionUtils::IEPrecisionToDataType(jcp.dst_prc));
    jcp.planar_layout = selectedPD->getConfig().inConfs[0].desc->checkGeneralLayout(GeneralLayout::ncsp);
    jcp.normalize_variance = normalizeVariance_;
    jcp.across_channels = acrossChannels_;
    int N = 0;
    std::tie(N, jcp.C, jcp.D, jcp.H, jcp.W) = shape5D;

    if (mayiuse(cpu::x64::avx512_common)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::avx512_common>(jcp, *attr.get()));

        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx512_common>(jcp));
        if (normalizeVariance_) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx512_common>(jcp));
        }
    } else if (mayiuse(cpu::x64::avx2)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::avx2>(jcp, *attr.get()));

        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx2>(jcp));
        if (normalizeVariance_) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx2>(jcp));
        }
    } else if (mayiuse(cpu::x64::sse41)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::sse41>(jcp, *attr.get()));

        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::sse41>(jcp));
        if (normalizeVariance_) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::sse41>(jcp));
        }
    }

    if (mvn_kernel)
        mvn_kernel->create_ker();

    if (mvn_mean_kernel)
        mvn_mean_kernel->create_ker();

    if (mvn_variance_kernel)
        mvn_variance_kernel->create_ker();
}

void MKLDNNMVNNode::transformTo5DCase(const SizeVector& shape) {
    switch (shape.size()) {
        // for 1 and 2 rank, if acrossChannels_ is true, adjust shape to fully vectorize under unified 5d procedure.
        // otherwise there are not enough data in spatial dimension to process in one kernel.
        case 1 :  // C
            if (acrossChannels_) {
                shape5D = std::make_tuple(1, 1, 1, 1, shape[0]);
                acrossChannels_ = false;
                break;
            } else {
                shape5D = std::make_tuple(1, shape[0], 1, 1, 1);
                break;
            }
        case 2 :  // NC
            if (acrossChannels_) {
                shape5D = std::make_tuple(1, shape[0], 1, shape[1], 1);
                acrossChannels_ = false;
                break;
            } else {
                shape5D = std::make_tuple(shape[0], shape[1], 1, 1, 1);
                break;
            }
        case 3 : { shape5D = std::make_tuple(shape[0], shape[1], 1, shape[2], 1); break; }
        case 4 : { shape5D = std::make_tuple(shape[0], shape[1], 1, shape[2], shape[3]); break; }
        case 5 : { shape5D = std::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4]); break; }
        default : { IE_THROW() << "MVN layer with name '" << getName() << "' doesn't support planar layout with rank: " << shape.size(); }
    }
}

void MKLDNNMVNNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
    mkldnn::post_ops ops;
    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops);
            continue;
        }
        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }
    attr.set_post_ops(ops);
}

void MKLDNNMVNNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();

    uint8_t *dst_data = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());
    uint8_t *src_data = reinterpret_cast<uint8_t*>(srcMemPtr->GetPtr());

    auto dim = getParentEdgeAt(0)->getShape().getStaticDims();
    if (mayiuse(cpu::x64::sse41)) {
        if (!mvn_mean_kernel || (normalizeVariance_ && !mvn_variance_kernel) || !mvn_kernel) {
            IE_THROW() << "MVN layer with name '" << getName() << "' doesn't create kernel to execute on sse41 above platform.";
        }
        if (getParentEdgeAt(0)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::ncsp)) {
            mvn_pln(src_data, dst_data, dim);
        } else {
            mvn_blk(src_data, dst_data, dim);
        }
    } else {
        mvn_ref(src_data, dst_data, dim);
    }
}

void MKLDNNMVNNode::mvn_pln(const uint8_t* src_data, uint8_t* dst_data, const SizeVector& dims) {
    size_t blk_size = 1;  // blk size in vmm
    if (mayiuse(cpu::x64::avx512_common)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 4;
    }

    size_t N = 0; size_t C = 0; size_t D = 0; size_t H = 0; size_t W = 0;
    std::tie(N, C, D, H, W) = shape5D;

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    size_t src_stride_size = static_cast<size_t>(blk_size * src_data_size);
    size_t dst_stride_size = static_cast<size_t>(blk_size * dst_data_size);

    for (size_t b = 0lu; b < N; b++) {
        size_t cb = b * C3;
        if (acrossChannels_) {
            // Calculate mean value for one instance in batch
            // Parallel sum for each channel
            float C3inv = 1.f / static_cast<float>(C3);
            float mean_temp = 0.0f;
            mean_temp = parallel_sum(C, mean_temp, [&](size_t c)->float {
                float mean_internal = 0.0f;
                size_t cc = cb + c * C2;
                auto arg = jit_mvn_call_args();
                arg.src = src_data + cc * src_data_size;
                arg.sum = static_cast<float*>(&mean_internal);
                arg.src_stride = src_stride_size;
                arg.work_amount = static_cast<size_t>(C2 / blk_size); // for vector part
                (*mvn_mean_kernel)(&arg);
                return mean_internal;
            });

            float mean = mean_temp * C3inv;

            // calculate variance value for one instance in batch
            // parallel sum for each channel
            if (normalizeVariance_) {
                float variance_temp = 0.0f;
                variance_temp = parallel_sum(C, variance_temp, [&](size_t c)->float {
                    float variance_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc * src_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance_internal);
                    arg.src_stride = src_stride_size;
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);  // vector part
                    (*mvn_variance_kernel)(&arg);
                    return variance_internal;
                });

                float variance = 1.f;
                if (epsMode_ == INSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C3inv + epsValue_);
                else if (epsMode_ == OUTSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C3inv) + epsValue_;
                // mvn for one instance in batch
                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc * src_data_size;
                    arg.dst = dst_data + cc * dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance);
                    arg.src_stride = src_stride_size;
                    arg.dst_stride = dst_stride_size;
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);  // work amount for vector part
                    arg.oc_off = sizeof(float) * c;
                    (*mvn_kernel)(&arg);
                });
            } else {
                // mvn for one instance in batch
                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc * src_data_size;
                    arg.dst = dst_data + cc * dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.src_stride = src_stride_size;
                    arg.dst_stride = dst_stride_size;
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);
                    arg.oc_off = sizeof(float) * c;
                    (*mvn_kernel)(&arg);
                });
            }
        } else {  // per channel
            float C2inv = 1.f / static_cast<float>(C2);
            parallel_for(C, [&](size_t c) {
                // mean for this channel
                float mean = 0.f;
                size_t cc = cb + c * C2;
                // the same arg for three kernels
                auto arg = jit_mvn_call_args();
                arg.src = src_data + cc * src_data_size;
                arg.dst = dst_data + cc * dst_data_size;
                arg.sum = static_cast<float*>(&mean);
                arg.src_stride = src_stride_size;
                arg.dst_stride = dst_stride_size;
                arg.work_amount = static_cast<size_t>(C2 / blk_size);
                arg.oc_off = static_cast<size_t>(c * sizeof(float));
                (*mvn_mean_kernel)(&arg);

                mean *= C2inv;

                if (normalizeVariance_) {
                    // variance for this channel
                    float variance = 0.f;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance);
                    (*mvn_variance_kernel)(&arg);

                    if (epsMode_ == INSIDE_SQRT)
                        variance = 1.f / sqrtf(variance * C2inv + epsValue_);
                    else if (epsMode_ == OUTSIDE_SQRT)
                        variance = 1.f / (sqrtf(variance * C2inv) + epsValue_);

                    // mvn for this channel
                    (*mvn_kernel)(&arg);
                } else {
                    // mvn for this channel
                    arg.mean = static_cast<float*>(&mean);
                    (*mvn_kernel)(&arg);
                }
            });
        }
    }
}

void MKLDNNMVNNode::mvn_ref(const uint8_t* src_data, uint8_t* dst_data, const SizeVector& dims) {
    const float *src_data_ptr = reinterpret_cast<const float *>(src_data);
    float *dst_data_ptr = reinterpret_cast<float *>(dst_data);
    size_t N = 0; size_t C = 0; size_t D = 0; size_t H = 0; size_t W = 0;
    std::tie(N, C, D, H, W) = shape5D;

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    for (size_t b = 0lu; b < N; b++) {
        size_t cb = b * C3;
        if (acrossChannels_) {
            // Parallel sum for each channel for mean
            float C3inv = 1.f / static_cast<float>(C3);
            float mean_temp = 0.0f;

            mean_temp = parallel_sum(C, mean_temp, [&](size_t c)->float {
                float mean_internal = 0.0f;
                size_t cc = cb + c * C2;
                for (size_t sp = 0lu; sp < C2; sp++) {
                    mean_internal += src_data_ptr[cc + sp];
                }
                return mean_internal;
            });

            float mean = mean_temp * C3inv;

            if (normalizeVariance_) {
                // parallel sum for each channel for variance
                float variance_temp = 0.0f;
                variance_temp = parallel_sum(C, variance_temp, [&](size_t c)->float {
                    float variance_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        variance_internal += (src_data_ptr[cc + sp] - mean) * (src_data_ptr[cc + sp] - mean);
                    }
                    return variance_internal;
                });

                float variance = 1.f;
                if (epsMode_ == INSIDE_SQRT)
                    variance = 1.f / sqrtf(variance_temp * C3inv + epsValue_);
                else if (epsMode_ == OUTSIDE_SQRT)
                    variance = 1.f / (sqrtf(variance_temp * C3inv) + epsValue_);

                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = (src_data_ptr[cc + sp] - mean) * variance;
                    }
                });
            } else {
                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = src_data_ptr[cc + sp] - mean;
                    }
                });
            }
        } else {  // per channel
            float C2inv = 1.f / static_cast<float>(C2);
            parallel_for(C, [&](size_t c) {
                // mean for this channel
                float mean = 0.f;
                size_t cc = cb + c * C2;
                for (size_t sp = 0lu; sp < C2; sp++) {
                    mean += src_data_ptr[cc + sp];
                }
                mean *= C2inv;

                if (normalizeVariance_) {
                    // variance for this channel
                    float variance = 0.f;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        variance += (src_data_ptr[cc + sp] - mean) * (src_data_ptr[cc + sp] - mean);
                    }

                    if (epsMode_ == INSIDE_SQRT)
                        variance = 1.f / sqrtf(variance * C2inv + epsValue_);
                    else if (epsMode_ == OUTSIDE_SQRT)
                        variance = 1.f / (sqrtf(variance * C2inv) + epsValue_);

                    // mvn for this channel
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = (src_data_ptr[cc + sp] - mean) * variance;
                    }
                } else {
                    // mvn for this channel
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = src_data_ptr[cc + sp] - mean;
                    }
                }
            });
        }
    }
}

void MKLDNNMVNNode::mvn_blk(const uint8_t* src_data, uint8_t* dst_data, const SizeVector& dims) {
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(cpu::x64::avx512_common)) {
        blk_size = 16;
    } else {
        blk_size = 8;
    }

    size_t N = 1; size_t C = 1; size_t D = 1; size_t H = 1; size_t W = 1;
    std::tie(N, C, D, H, W) = shape5D;

    bool is_nhwc = getParentEdgeAt(0)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::nspc);

    size_t CB = div_up(C, blk_size);

    size_t C0 = is_nhwc ? W * C : W * blk_size;
    size_t C1 = C0 * H;
    size_t C2 = C1 * D;
    size_t C3 = C2 * CB;
    size_t C5 = C * D * H * W;

    size_t threads_num = parallel_get_num_threads();
    size_t aux_buffer_size = acrossChannels_ ? blk_size : rnd_up(C, blk_size);
    std::vector<float> mean_buffer(aux_buffer_size * threads_num);
    std::vector<float> variance_buffer(aux_buffer_size * threads_num);

    size_t src_stride_size = is_nhwc ? static_cast<size_t>(C * src_data_size) : static_cast<size_t>(blk_size * src_data_size);
    size_t dst_stride_size = is_nhwc ? static_cast<size_t>(C * dst_data_size) : static_cast<size_t>(blk_size * dst_data_size);

    for (size_t b = 0lu; b < N; b++) {
        size_t b_offset = is_nhwc ? b * C5 : b * C3;
        if (acrossChannels_) {
            // mean for this instance in batch
            float C5inv = 1.f / static_cast<float>(C5);
            float mean_temp = 0.0f;
            mean_temp = parallel_sum3d(CB, D, H, mean_temp, [&](size_t cb, size_t d, size_t h)->float {
                size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                            : b_offset + cb * C2 + d * C1 + h * C0;

                float mean_internal = 0.0f;
                /////////////////////////////////
                //          W           //  |
                //                      //  |
                //                      //  |
                //blk +  +  +  +  +  +  //  |  +
                //                      //  |
                //                      //  |
                //                      // \|/
                /////////////////////////////////
                auto mean_buffer_ptr = &mean_buffer[blk_size * parallel_get_thread_num()];
                for (int i = 0; i < blk_size; i++)
                    mean_buffer_ptr[i] = 0.f;

                auto arg = jit_mvn_call_args();
                arg.src = src_data + src_offset * src_data_size;
                arg.sum = mean_buffer_ptr;
                arg.src_stride = src_stride_size;
                arg.work_amount = static_cast<size_t>(W);
                arg.oc_off = static_cast<size_t>(cb * blk_size * sizeof(float));  // for tail process
                (*mvn_mean_kernel)(&arg);  // for W * blk

                size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                for (int i = 0; i < min_cb; i++)
                    mean_internal += mean_buffer_ptr[i];
                return mean_internal;
            });
            float mean = mean_temp * C5inv;

            if (normalizeVariance_) {
                // variance: sum((x-mean)*(x-mean)) for one instance in batch
                float variance_temp = 0.0f;
                variance_temp = parallel_sum3d(CB, D, H, variance_temp, [&](size_t cb, size_t d, size_t h)->float {
                    size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                : b_offset + cb * C2 + d * C1 + h * C0;

                    float variance_internal = 0.0f;
                    auto variance_buffer_ptr = &variance_buffer[blk_size * parallel_get_thread_num()];
                    for (int i = 0; i < blk_size; i++)
                        variance_buffer_ptr[i] = 0.f;

                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * src_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = variance_buffer_ptr;
                    arg.src_stride = src_stride_size;
                    arg.work_amount = static_cast<size_t>(W);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    (*mvn_variance_kernel)(&arg);

                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                    for (int i = 0; i < min_cb; i++)
                        variance_internal += variance_buffer_ptr[i];
                    return variance_internal;
                });

                float variance = 1.f;
                if (epsMode_ == INSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C5inv + epsValue_);
                else if (epsMode_ == OUTSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C5inv) + epsValue_;
                // mvn for one instance in batch
                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                : b_offset + cb * C2 + d * C1 + h * C0;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * src_data_size;
                    arg.dst = dst_data + src_offset * dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance);
                    arg.src_stride = src_stride_size;
                    arg.dst_stride = dst_stride_size;
                    arg.work_amount = static_cast<size_t>(W);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    (*mvn_kernel)(&arg);
                });
            } else {
                // mvn for one instance in batch
                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                : b_offset + cb * C2 + d * C1 + h * C0;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * src_data_size;
                    arg.dst = dst_data + src_offset * dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.src_stride = src_stride_size;
                    arg.dst_stride = dst_stride_size;
                    arg.work_amount = static_cast<size_t>(W);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    (*mvn_kernel)(&arg);
                });
            }
        } else {  // for per_channel
            float size_inv = 1.f / static_cast<float>(D * H * W);
            for (int i = 0; i < mean_buffer.size(); i++)
                mean_buffer[i] = 0.f;

            // one thread for one C*W size(the same H) to get C size result for the same H, added to last group result
            // keep the compute order the same as planar
            parallel_for2d(D, H, [&](size_t thr_idx, size_t d, size_t h) {
                for (size_t cb = 0; cb < CB; cb++) {
                    size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                : b_offset + cb * C2 + d * C1 + h * C0;
                    auto mean_buffer_ptr = &mean_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * src_data_size;
                    arg.sum = mean_buffer_ptr;
                    arg.src_stride = src_stride_size;
                    arg.work_amount = static_cast<size_t>(W);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    (*mvn_mean_kernel)(&arg);
                }
            });

            for (size_t i = 1; i < threads_num; i++) {
                for (size_t c = 0; c < C; c++)
                    mean_buffer[c] += mean_buffer[c + aux_buffer_size * i];
            }
            for (size_t c = 0; c < C; c++)
                mean_buffer[c] *= size_inv;

            if (normalizeVariance_) {
                for (int i = 0; i < variance_buffer.size(); i++)
                    variance_buffer[i] = 0.f;

                parallel_for2d(D, H, [&](size_t thr_idx, size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                    : b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_offset * src_data_size;
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.src_stride = src_stride_size;
                        arg.work_amount = static_cast<size_t>(W);
                        arg.oc_off = cb * blk_size * sizeof(float);
                        (*mvn_variance_kernel)(&arg);
                    }
                });
                for (size_t i = 1; i < threads_num; i++) {
                    for (size_t c = 0; c < C; c++)
                        variance_buffer[c] += variance_buffer[c + aux_buffer_size * i];
                }
                for (size_t c = 0; c < C; c++) {
                    if (epsMode_ == INSIDE_SQRT)
                        variance_buffer[c] = 1.f / sqrtf(variance_buffer[c] * size_inv + epsValue_);
                    else if (epsMode_ == OUTSIDE_SQRT)
                        variance_buffer[c] = 1.f / (sqrtf(variance_buffer[c] * size_inv) + epsValue_);
                }

                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                    : b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_offset * src_data_size;
                        arg.dst = dst_data + src_offset * dst_data_size;
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.src_stride = src_stride_size;
                        arg.dst_stride = dst_stride_size;
                        arg.work_amount = static_cast<size_t>(W);
                        arg.oc_off = cb * blk_size * sizeof(float);
                        (*mvn_kernel)(&arg);
                    }
                });
            } else {
                // normalizeVariance_ == false
                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                    : b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_offset * src_data_size;
                        arg.dst = dst_data + src_offset * dst_data_size;
                        arg.mean = mean_buffer_ptr;
                        arg.src_stride = src_stride_size;
                        arg.dst_stride = dst_stride_size;
                        arg.work_amount = static_cast<size_t>(W);
                        arg.oc_off = cb * blk_size * sizeof(float);
                        (*mvn_kernel)(&arg);
                    }
                });
            }
        }
    }
}

bool MKLDNNMVNNode::canFuse(const MKLDNNNodePtr& node) const {
    if (!mayiuse(cpu::x64::sse41)) {
        return false;
    }
    // limit post ops to unary when shape transformed on channel
    // 1D only fused with unary
    int inputRank = getParentEdgeAt(0)->getShape().getRank();
    bool unaryEltwise = one_of(node->getAlgorithm(), EltwiseRelu, EltwiseGelu, EltwiseElu, EltwiseSigmoid, EltwiseClamp, EltwiseTanh,
                                            EltwiseSwish, EltwiseHswish, EltwiseMish, EltwiseHsigmoid, EltwiseRoundHalfToEven,
                                            EltwiseRoundHalfAwayFromZero, EltwiseAbs, EltwiseSqrt, EltwiseSoftRelu);
    if ((inputRank == 1 && !unaryEltwise) ||
        (inputRank == 2 && !unaryEltwise && acrossChannels_)) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool MKLDNNMVNNode::created() const {
    return getType() == MVN;
}

REG_MKLDNN_PRIM_FOR(MKLDNNMVNNode, MVN);
