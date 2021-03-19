// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_mvn_node.h"

#include "mkldnn_quantize_node.h"
#include "mkldnn_eltwise_node.h"
#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "utils/bfloat16.hpp"
#include <legacy/ie_layers_internal.hpp>
#include "ie_parallel.hpp"
#include <algorithm>
#include "emitters/jit_load_store_emitters.hpp"
#include "emitters/jit_bf16_emitters.hpp"

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/jit_uni_eltwise.hpp>
#include <cpu/x64/jit_uni_depthwise_injector.hpp>
#include <cpu/x64/jit_uni_quantization_injector.hpp>
#include <cpu/x64/jit_uni_eltwise_injector.hpp>
#include <mkldnn_selective_build.h>
#include "emitters/jit_eltwise_emitters.hpp"
#include "emitters/jit_mkldnn_emitters.hpp"

#include <ngraph/opsets/opset6.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_mvn_call_args, field)

namespace {

struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    jit_generator *host;
    cpu_isa_t host_isa;
    const MKLDNNNode *node;
    InferenceEngine::Precision exec_prc;
};

template<typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext & ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.host_isa, ctx.node, ctx.exec_prc);
    }
};

}   // namespace

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
        mov(reg_src, ptr[reg_params + GET_OFF(src[0])]);
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
                    mov(reg_src, ptr[reg_params + GET_OFF(src[0])]);
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
                            std::make_shared<load_emitter_context>(jcp_.src_prc, dst_prc, tail_num, true, "zero"),
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

    explicit jit_uni_mvn_kernel_f32(jit_mvn_config_params jcp, MKLDNNMVNNode& node) : jit_uni_mvn_kernel(jcp, node), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        Precision exec_prc = Precision::FP32;
        for (int i = 0; i < MVNNode.getFusedWith().size(); i++) {
            if (MVNNode.isDepthWiseNode(MVNNode.getFusedWith()[i])) {
                auto* depthwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(MVNNode.getFusedWith()[i].get());
                depthwiseNode->appendPostOps(post_ops);
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                    this,
                    post_ops.get()->entry_[post_ops.len() - 1].depthwise.alg));
            } else if (MVNNode.getFusedWith()[i].get()->getType() == Eltwise) {
                post_op_emitters.push_back(create_eltwise_emitter(*MVNNode.getFusedWith()[i].get(), exec_prc));
            } else if (MVNNode.getFusedWith()[i].get()->getType() == Quantize) {
                auto quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(MVNNode.getFusedWith()[i].get());
                quantizeNode->appendPostOps(post_ops);

                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_ops.get()->entry_[post_ops.len() - 1], vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        load_emitter.reset(new jit_load_emitter(this, isa, nullptr));
        store_emitter.reset(new jit_store_emitter(this, isa, nullptr));

        this->preamble();

        for (int i = 0; i < jcp_.inputs_number; i++) {
            mov(get_src_reg(i), ptr[reg_params + GET_OFF(src[0]) + i * sizeof(size_t)]);
        }
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
                    for (int i = 0; i < jcp_.inputs_number; i++) {
                        mov(get_src_reg(i), ptr[reg_params + GET_OFF(src[0]) + i * sizeof(size_t)]);
                    }
                    mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
                    mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

                    for (int i = 0; i < jcp_.inputs_number; i++) {
                        add(get_src_reg(i), offset_sse42 * jcp_.src_data_size);
                    }
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
        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core) && store_emitter != nullptr && store_emitter->get_emu_vcvtneps2bf16() != nullptr)
            store_emitter->get_emu_vcvtneps2bf16()->emit_data();

        for (int i = 0; i < post_op_emitters.size(); i++) {
            post_op_emitters[i]->emit_data();
        }
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
    Xbyak::Reg64 reg_load_store_mask = rcx;

    Xbyak::Reg64 reg_post_src0 = rsi;
    Xbyak::Reg64 reg_post_src1 = rbp;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_mean = Vmm(1);
    Vmm vmm_variance_inv = Vmm(2);
    Vmm vmm_zero = Vmm(3);

    Vmm vmm_d_weights = Vmm(5);
    Vmm vmm_d_bias = Vmm(6);

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;

    mkldnn::post_ops post_ops;
    std::vector<std::shared_ptr<jit_emitter>> post_op_emitters = {};
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors = {};
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors = {};

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;
    std::vector<size_t> load_pool_gpr_idxs;

    Reg64 get_src_reg(int idx) {
        switch (idx) {
            case 0: return reg_src; break;
            case 1: return reg_post_src0; break;
            case 2: return reg_post_src1; break;
            default: IE_THROW() << "MVN layer have unsupported number of input";
        }
    }

    Vmm get_src_vmm(int idx) {
        switch (idx) {
            case 0: return vmm_val; break;
            case 1: return Vmm(15); break;
            case 2: return Vmm(14); break;
            default: IE_THROW() << "MVN layer have unsupported number of input";
        }
    }

    Vmm get_aux_vmm(int idx) {
        if (idx + 8 >= 13)
            IE_THROW() << "MVN layer need unsupported number of aux vmm";
        return Vmm(idx + 8);
    }

    inline void worker_mvn(bool is_tail) {
        int elt_num = is_tail ? tail_num : step;
        for (int i = 0; i < jcp_.inputs_number; i++) {
            load_emitter->emit_code({static_cast<size_t>(get_src_reg(i).getIdx())}, {static_cast<size_t>(get_src_vmm(i).getIdx())},
                std::make_shared<load_emitter_context>(jcp_.src_prc, Precision::FP32, elt_num),
                {}, {load_pool_gpr_idxs});
        }

        uni_vsubps(vmm_val, vmm_val, vmm_mean);
        if (jcp_.normalize_variance)
            uni_vmulps(vmm_val, vmm_val, vmm_variance_inv);

        apply_post_ops(jcp_.planar_layout);

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

            for (int i = 0; i < jcp_.inputs_number; i++) {
                add(get_src_reg(i), reg_src_stride);
            }
            add(reg_dst, reg_dst_stride);
            sub(reg_work_amount, 1);

            jmp(mvn_loop_label, T_NEAR);
        }
        L(mvn_loop_end_label);
    }

    inline void apply_post_ops(bool is_broadcast) {
        int src_idx = 0;
        int eltwise_post_op_idx = 0;
        int depthwise_post_op_idx = 0;
        int quantization_post_op_idx = 0;
        int injector_idx = 0;
        for (int i = 0; i < MVNNode.getFusedWith().size(); i++) {
            if (MVNNode.isDepthWiseNode(MVNNode.getFusedWith()[i])) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_ops.get()->entry_[injector_idx].depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_ops.get()->entry_[injector_idx].depthwise.biases_data));
                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);
                // weight and bias is padded. scalar as vector.
                depthwise_injectors[depthwise_post_op_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1,
                    reg_d_weights, reg_d_bias, is_broadcast);
                depthwise_post_op_idx++;
                injector_idx++;
            } else if (MVNNode.getFusedWith()[i].get()->getType() == Eltwise) {
                std::vector<size_t> in_idxs;
                std::vector<size_t> aux_idxs;
                in_idxs.push_back(vmm_val.getIdx());
                for (int j = 1; j < post_op_emitters[eltwise_post_op_idx]->get_inputs_num(); j++)
                    in_idxs.push_back(get_src_vmm(++src_idx).getIdx());
                for (int j = 0; j < post_op_emitters[eltwise_post_op_idx]->aux_vecs_count(); j++)
                    aux_idxs.push_back(get_aux_vmm(j).getIdx());

                std::vector<size_t> out_idxs;
                out_idxs.push_back(vmm_val.getIdx());

                post_op_emitters[eltwise_post_op_idx]->emit_code(in_idxs, out_idxs, aux_idxs);

                eltwise_post_op_idx++;
            } else {
                auto quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(MVNNode.getFusedWith()[i].get());

                bool do_dequantization = quantizeNode->getOpType() == QuantizeOpType::FakeQuantization;
                bool do_rounding = do_dequantization || isFloatCompatible(jcp_.dst_prc) || i != MVNNode.getFusedWith().size() - 1;
                int s_idx = vmm_val.getIdx();

                quantization_injectors[quantization_post_op_idx]->init_crop_ptrs(reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_post_op_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding,
                                                                                            0, is_broadcast);

                quantization_injectors[quantization_post_op_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_post_op_idx++;
                injector_idx++;
            }
        }
    }

    std::shared_ptr<jit_emitter> create_eltwise_emitter(MKLDNNNode& node, Precision exec_prec) {
        const auto& eltwiseNode = dynamic_cast<const MKLDNNEltwiseNode&>(node);

        EltwiseEmitterContext ctx = {
            nullptr,
            this,
            isa,
            &node,
            exec_prec
        };

        OV_SWITCH(MKLDNNPlugin, EltwiseEmitter, ctx, eltwiseNode.getOpType(),
        OV_CASE(Relu, jit_mkldnn_aux_emitter),
        OV_CASE(Gelu, jit_mkldnn_aux_emitter),
        OV_CASE(Elu, jit_mkldnn_aux_emitter),
        OV_CASE(Tanh, jit_mkldnn_aux_emitter),
        OV_CASE(Logistic, jit_mkldnn_aux_emitter),
        OV_CASE(Square, jit_mkldnn_aux_emitter),
        OV_CASE(Abs, jit_mkldnn_aux_emitter),
        OV_CASE(Sqrt, jit_mkldnn_aux_emitter),
        OV_CASE(Linear, jit_mkldnn_aux_emitter),
        OV_CASE(BoundedRelu, jit_mkldnn_aux_emitter),
        OV_CASE(SoftRelu, jit_mkldnn_aux_emitter),
        OV_CASE(Relu6, jit_mkldnn_aux_emitter),
        OV_CASE(Exp, jit_mkldnn_aux_emitter),
        OV_CASE(Clamp, jit_mkldnn_aux_emitter),
        OV_CASE(Swish, jit_mkldnn_aux_emitter),
        OV_CASE(Hswish, jit_mkldnn_aux_emitter),
        OV_CASE(Mish, jit_mkldnn_aux_emitter),
        OV_CASE(Hsigmoid, jit_mkldnn_aux_emitter),
        OV_CASE(Round, jit_mkldnn_aux_emitter),
        OV_CASE(Add, jit_add_emitter),
        OV_CASE(MulAdd, jit_mul_add_emitter),
        OV_CASE(Subtract, jit_subtract_emitter),
        OV_CASE(Multiply, jit_multiply_emitter),
        OV_CASE(Divide, jit_divide_emitter),
        OV_CASE(FloorMod, jit_floor_mod_emitter),
        OV_CASE(Mod, jit_mod_emitter),
        OV_CASE(Maximum, jit_maximum_emitter),
        OV_CASE(Minimum, jit_minimum_emitter),
        OV_CASE(SquaredDifference, jit_squared_difference_emitter),
        OV_CASE(PowerDynamic, jit_power_dynamic_emitter),
        OV_CASE(Equal, jit_equal_emitter),
        OV_CASE(NotEqual, jit_not_equal_emitter),
        OV_CASE(Greater, jit_greater_emitter),
        OV_CASE(GreaterEqual, jit_greater_equal_emitter),
        OV_CASE(Less, jit_less_emitter),
        OV_CASE(LessEqual, jit_less_equal_emitter),
        OV_CASE(LogicalAnd, jit_logical_and_emitter),
        OV_CASE(LogicalOr, jit_logical_or_emitter),
        OV_CASE(LogicalXor, jit_logical_xor_emitter),
        OV_CASE(LogicalNot, jit_logical_not_emitter),
        OV_CASE(PowerStatic, jit_power_static_emitter),
        OV_CASE(Prelu, jit_prelu_emitter));

        if (!ctx.emitter)
            IE_THROW() << "Unsupported operation type for Eltwise emitter";

        return ctx.emitter;
    }
};
//////////////////////////////////////////////////////////////////////////////////

MKLDNNMVNNode::MKLDNNMVNNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache), epsMode_(insideSqrt) {}

void MKLDNNMVNNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    std::string errPrefix = "MVN node with name '" + getName() + "' ";

    auto cnnLayer = getCnnLayer();
    if (cnnLayer == nullptr)
        IE_THROW() << errPrefix << "does not have CNN layer.";

    if (getChildEdges().empty())
        IE_THROW() << errPrefix << "has incorrect number of output edges.";

    const auto& numOfDims = getParentEdgeAt(0)->getDims().ndims();
    if (numOfDims < 1 || numOfDims > 5)
        IE_THROW() << errPrefix << "doesn't support input with size of dimensions: " << numOfDims;

    across_channels = false;
    if (!hasAxesInput) {
        across_channels = cnnLayer->GetParamAsBool("across_channels");
    } else {
        if (numOfDims == getParentEdgeAt(1)->getDims().size() + 1 || numOfDims == 1)
            across_channels = true;
    }
    normalize_variance = cnnLayer->GetParamAsBool("normalize_variance", true);
    eps = cnnLayer->GetParamAsFloat("eps");
    auto epsMode = cnnLayer->GetParamAsString("eps_mode", "");
    if (details::CaselessEq<std::string>()(epsMode, "inside_sqrt")) {
        epsMode_ = insideSqrt;
    } else if (details::CaselessEq<std::string>()(epsMode, "outside_sqrt")) {
        epsMode_ = outsideSqrt;
    }
}

void MKLDNNMVNNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<Precision> supportedPrecisions = {
            Precision::FP32,
            Precision::U8,
            Precision::I8,
            Precision::U16,
            Precision::I16,
            Precision::BF16,
            Precision::I32
    };
    auto filterPrecision = [&](Precision& prc) {
        if (!mayiuse(cpu::x64::sse41)) {
            return Precision(Precision::FP32);
        } else if (std::find(supportedPrecisions.begin(), supportedPrecisions.end(), prc) == supportedPrecisions.end()) {
            if (prc == Precision::U32 || prc == Precision::I64 || prc == Precision::U64) {
                return Precision(Precision::I32);
            } else {
                IE_THROW() << "MVN node with name `" << getName() << "` doesn't support " << prc << " precision.";
            }
        } else {
            return prc;
        }
    };

    Precision inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    inputPrecision = filterPrecision(inputPrecision);
    if (getParentEdgeAt(0)->getDims().ndims() < 3 || getParentEdgeAt(0)->getDims().ndims() > 5
        || across_channels != 0 || normalize_variance != 1) {
        if (!isFloatCompatible(inputPrecision)) {
            inputPrecision = Precision::FP32;
        }
    }

    Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();
    outputPrecision = filterPrecision(outputPrecision);
    if (!mayiuse(avx512_core)) {
        if (outputPrecision == Precision::BF16)
            outputPrecision = Precision::FP32;
    }

    if (!fusedWith.empty()) {
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputPrecision = lastFusedLayer->outData[0]->getPrecision();
        }
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    input_prec = inputPrecision;
    output_prec = outputPrecision;
    src_data_size = MKLDNNExtensionUtils::sizeOfDataType(inputDataType);
    dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(outputDataType);

    size_t expectedInputsNum = hasAxesInput ? 2 : 1;
    for (auto& postOp : fusedWith) {
        auto* eltwiseNode = dynamic_cast<const MKLDNNEltwiseNode*>(postOp.get());
        if (eltwiseNode != nullptr && !isDepthWiseNode(postOp)) {
            expectedInputsNum += eltwiseNode->getOpInputsNum() - 1;
        }
    }
    int maxMVNAllInput = hasAxesInput ? MAX_MVN_INPUTS + 1 : MAX_MVN_INPUTS;
    if (getParentEdges().size() > maxMVNAllInput)
        IE_THROW() << "MVN node with name `" << getName() << "` doesn't support more than " << maxMVNAllInput
                           << " inputs (actual = " << getParentEdges().size() << ")";

    if (expectedInputsNum != getParentEdges().size())
        IE_THROW() << "MVN node with name `" << getName() << "` has invalid input number of inputs: expected = " << expectedInputsNum
                           << " (actual = " << getParentEdges().size() << ")";

    bool canBeInplace = (getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1) &&
                        !getParentEdgeAt(0)->getParent()->isConstant() &&
                        !getParentEdgeAt(0)->getChild()->isConstant() &&
                        (inputPrecision == outputPrecision);

    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };

    auto initDesc = [&] (LayoutType lt) -> PrimitiveDescInfo {
        auto createMemoryDesc = [lt](MKLDNNEdgePtr edge, Precision prc, size_t offset) -> TensorDesc {
            if (lt == ChannelsFirst) {
                auto dims = edge->getDims().ToSizeVector();
                auto ndims = dims.size();
                std::vector<size_t> order(ndims);
                std::iota(order.begin(), order.end(), 0);
                if (ndims > 1) {
                    order.erase(order.begin() + 1);
                    order.push_back(1);
                }

                std::vector<size_t> blocks(ndims);
                for (size_t i = 0; i < order.size(); i++) {
                    blocks[i] = dims[order[i]];
                }

                return TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset});
            } else if (lt == Blocked && edge->getDims()[1] != 1) {
                size_t blockSize = mayiuse(cpu::x64::avx512_common) ? 16 : 8;

                std::vector<size_t> blocks = edge->getDims().ToSizeVector();
                std::vector<size_t> order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = div_up(blocks[1], blockSize);
                blocks.push_back(blockSize);
                order.push_back(1);

                return TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset});
            } else {
                std::vector<size_t> blocks = edge->getDims().ToSizeVector();
                std::vector<size_t> order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset});
            }
        };

        size_t offset = std::numeric_limits<size_t>::max();
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = getChildEdgeAt(0)->getDims().ndims() > 1 && getChildEdgeAt(0)->getDims() == getParentEdgeAt(0)->getDims();

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = (!i && canBeInplace) ? 0 : -1;
            dataConfig.constant = false;

            if (hasAxesInput && i == 1) {
                dataConfig.constant = true;
                const auto& dims = getCnnLayer()->insData[1].lock()->getTensorDesc().getDims();
                dataConfig.desc = TensorDesc(Precision::I32, dims, TensorDesc::getLayoutByDims(dims));
            } else {
                dataConfig.constant = false;
                dataConfig.desc = createMemoryDesc(getParentEdgeAt(i), inputPrecision, offset);
            }

            config.inConfs.push_back(dataConfig);
        }

        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        dataConfig.desc = createMemoryDesc(getChildEdgeAt(0), outputPrecision, offset);

        config.outConfs.push_back(dataConfig);

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

        return {config, impl_type};
    };

    if (mayiuse(cpu::x64::sse41)) {
        auto ndim = getParentEdgeAt(0)->getDims().ndims();
        // nspc and cBlk
        if (ndim == 4 || ndim == 5) {
            supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
            supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
        }
    }
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}

void MKLDNNMVNNode::selectOptimalPrimitiveDescriptor() {
    for (auto& type : getPrimitivesPriority()) {
        int selectedPrimitive = -1;
        int equalsFormatCount = -1;
        for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {
            impl_desc_type supportedType = getSupportedPrimitiveDescriptors()[i].getImplementationType();
            if (type == supportedType) {
                int equalsLocalFormatCount = 0;
                if (getSupportedPrimitiveDescriptors()[i].getConfig().inConfs.size() > getParentEdges().size())
                    continue;
                for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[i].getConfig().inConfs.size(); j++) {
                    auto parentEdge = getParentEdgeAt(j);
                    auto parentPtr = parentEdge->getParent();
                    // We don't take into account constant edges since reorders on them will be executed on load network stage
                    if (j > 0 && parentPtr->isConstant()) {
                        equalsLocalFormatCount++;
                        continue;
                    }

                    auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

                    if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
                        int inNum = parentEdge->getInputNum();
                        if (inNum < 0 || inNum >= parent_spd->getConfig().outConfs.size()) {
                            inNum = 0;
                        }
                        if (MKLDNNExtensionUtils::initTensorsAreEqual(
                                getSupportedPrimitiveDescriptors()[i].getConfig().inConfs[j].desc,
                                parent_spd->getConfig().outConfs[inNum].desc)) {
                            equalsLocalFormatCount++;
                        }
                    }
                }
                if (equalsLocalFormatCount > equalsFormatCount) {
                    equalsFormatCount = equalsLocalFormatCount;
                    selectedPrimitive = static_cast<int>(i);
                }
            }
        }
        if (selectedPrimitive >= 0) {
            selectPrimitiveDescriptorByIndex(selectedPrimitive);
            return;
        }
    }

    if (getSupportedPrimitiveDescriptors().empty())
        IE_THROW() << "Supported primitive descriptors list is empty for node: " << getName();
    // fallback. If there are no primitives from priority list just select a first
    selectPrimitiveDescriptorByIndex(0);
}

void MKLDNNMVNNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (!isInitConfig(config)) {
        for (size_t i = 0; i < config.inConfs.size(); i++) {
            config.inConfs[i].desc = getConfiguredInputDesc(config, i);
        }

        for (size_t i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].desc = getConfiguredOutputDesc(config, i);
        }

        initDescriptor(config);
    } else {
        initDescriptor(config);
    }
}

std::tuple<size_t, size_t, size_t, size_t, size_t> MKLDNNMVNNode::get5dShapes(const SizeVector& dims) {
    std::tuple<size_t, size_t, size_t, size_t, size_t> shapes;
    switch (dims.size()) {
        case 1 : { shapes = std::make_tuple(1, dims[0], 1, 1, 1); break; }
        case 2 : { shapes = std::make_tuple(dims[0], dims[1], 1, 1, 1); break; }
        case 3 : { shapes = std::make_tuple(dims[0], dims[1], 1, dims[2], 1); break; }
        case 4 : { shapes = std::make_tuple(dims[0], dims[1], 1, dims[2], dims[3]); break; }
        case 5 : { shapes = std::make_tuple(dims[0], dims[1], dims[2], dims[3], dims[4]); break; }
        default : { IE_THROW() << "MVN layer with name '" << getCnnLayer()->name << "' doesn't support planar layout with rank: " << dims.size(); }
    }
    return shapes;
}

void MKLDNNMVNNode::createPrimitive() {
    size_t inputNum = getParentEdges().size();
    for (size_t i = 0; i < inputNum; ++i) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            IE_THROW() << "MVN layer with name '" << getCnnLayer()->name << "' didn't allocate input memory.";
    }
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "MVN layer with name '" << getCnnLayer()->name << "' didn't allocate destination memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "MVN layer with name '" << getCnnLayer()->name << "' didn't set preferable primitive descriptor.";

    start_offset_in.resize(inputNum);
    for (size_t i = 0; i < inputNum; i++) {
        start_offset_in[i] = getParentEdgeAt(i)->getMemory().GetDescriptor().data.offset0 *
                           MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(getParentEdgeAt(i)->getMemory().GetDescriptor().data.data_type));
    }
    start_offset_out = getChildEdgeAt(0)->getMemory().GetDescriptor().data.offset0 *
                     MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(getChildEdgeAt(0)->getMemory().GetDescriptor().data.data_type));

    auto selectedPD = getSelectedPrimitiveDescriptor();
    auto jcp = jit_mvn_config_params();
    jcp.inputs_number = hasAxesInput ? inputNum - 1 : inputNum;
    jcp.src_prc = selectedPD->getConfig().inConfs[0].desc.getPrecision();
    jcp.dst_prc = selectedPD->getConfig().outConfs[0].desc.getPrecision();
    jcp.src_data_size = MKLDNNExtensionUtils::sizeOfDataType(MKLDNNExtensionUtils::IEPrecisionToDataType(jcp.src_prc));
    jcp.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(MKLDNNExtensionUtils::IEPrecisionToDataType(jcp.dst_prc));
    jcp.planar_layout = MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims()) == selectedPD->getConfig().inConfs[0].desc.getLayout();
    jcp.normalize_variance = normalize_variance;
    jcp.across_channels = across_channels;
    SizeVector in_dims = getParentEdgeAt(0)->getDims().ToSizeVector();
    int N = 0;
    std::tie(N, jcp.C, jcp.D, jcp.H, jcp.W) = get5dShapes(in_dims);

    if (mayiuse(cpu::x64::avx512_common)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::avx512_common>(jcp, *this));

        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx512_common>(jcp));
        if (normalize_variance) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx512_common>(jcp));
        }
    } else if (mayiuse(cpu::x64::avx2)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::avx2>(jcp, *this));

        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx2>(jcp));
        if (normalize_variance) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx2>(jcp));
        }
    } else if (mayiuse(cpu::x64::sse41)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::sse41>(jcp, *this));

        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::sse41>(jcp));
        if (normalize_variance) {
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

void MKLDNNMVNNode::execute(mkldnn::stream strm) {
    size_t inputNum = getParentEdges().size();
    int dataInputNum = hasAxesInput ? inputNum - 1 : inputNum;
    std::vector<const uint8_t *> src_data(dataInputNum);
    for (int i = 0; i < dataInputNum; i++) {
        int edgeInx = i;
        if (hasAxesInput && i != 0) {
            edgeInx++;
        }
        src_data[i] = reinterpret_cast<const uint8_t*>(getParentEdgeAt(edgeInx)->getMemory().GetData()) + start_offset_in[edgeInx];
    }
    uint8_t *dst_data = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemory().GetData()) + start_offset_out;

    auto dim = getParentEdgeAt(0)->getDesc().getDims();
    if (mayiuse(cpu::x64::sse41)) {
        if (!mvn_mean_kernel || (normalize_variance && !mvn_variance_kernel) || !mvn_kernel) {
            IE_THROW() << "MVN layer with name '" << getCnnLayer()->name << "' doesn't create kernel to execute on sse41 above platform.";
        }
        Layout layout = getParentEdgeAt(0)->getDesc().getLayout();
        if (layout == C || layout == NC || layout == CHW || layout == NCHW || layout == NCDHW) {
            mvn_pln(src_data, dst_data, dim);
        } else {
            mvn_blk(src_data, dst_data, dim);
        }
    } else {
        mvn_ref(src_data[0], dst_data, dim);
    }
}

void MKLDNNMVNNode::mvn_pln(const std::vector<const uint8_t *>& src_data, uint8_t* dst_data, const SizeVector& dims) {
    size_t input_num = src_data.size();
    size_t blk_size = 1;  // blk size in vmm
    if (mayiuse(cpu::x64::avx512_common)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 4;
    }

    size_t N = 0; size_t C = 0; size_t D = 0; size_t H = 0; size_t W = 0;
    std::tie(N, C, D, H, W) = get5dShapes(dims);

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    size_t src_stride_size = static_cast<size_t>(blk_size * src_data_size);
    size_t dst_stride_size = static_cast<size_t>(blk_size * dst_data_size);

    for (size_t b = 0lu; b < N; b++) {
        size_t cb = b * C3;
        if (across_channels) {
            // Calculate mean value for one instance in batch
            // Parallel sum for each channel
            float C3inv = 1.f / static_cast<float>(C3);
            float mean_temp = 0.0f;
            mean_temp = parallel_sum(C, mean_temp, [&](size_t c)->float {
                float mean_internal = 0.0f;
                size_t cc = cb + c * C2;
                auto arg = jit_mvn_call_args();
                arg.src[0] = src_data[0] + cc * src_data_size;
                arg.sum = static_cast<float*>(&mean_internal);
                arg.src_stride = src_stride_size;
                arg.work_amount = static_cast<size_t>(C2 / blk_size); // for vector part
                (*mvn_mean_kernel)(&arg);
                return mean_internal;
            });

            float mean = mean_temp * C3inv;

            // calculate variance value for one instance in batch
            // parallel sum for each channel
            if (normalize_variance) {
                float variance_temp = 0.0f;
                variance_temp = parallel_sum(C, variance_temp, [&](size_t c)->float {
                    float variance_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src[0] = src_data[0] + cc * src_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance_internal);
                    arg.src_stride = src_stride_size;
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);  // vector part
                    (*mvn_variance_kernel)(&arg);
                    return variance_internal;
                });

                float variance = 1.f;
                if (epsMode_ == insideSqrt)
                    variance /= sqrtf(variance_temp * C3inv + eps);
                else if (epsMode_ == outsideSqrt)
                    variance /= sqrtf(variance_temp * C3inv) + eps;
                // mvn for one instance in batch
                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    for (int i = 0; i < input_num; ++i)
                        arg.src[i] = src_data[i] + cc * src_data_size;
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
                    for (int i = 0; i < input_num; ++i)
                        arg.src[i] = src_data[i] + cc * src_data_size;
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
                for (int i = 0; i < input_num; ++i)
                    arg.src[i] = src_data[i] + cc * src_data_size;
                arg.dst = dst_data + cc * dst_data_size;
                arg.sum = static_cast<float*>(&mean);
                arg.src_stride = src_stride_size;
                arg.dst_stride = dst_stride_size;
                arg.work_amount = static_cast<size_t>(C2 / blk_size);
                arg.oc_off = static_cast<size_t>(c * sizeof(float));
                (*mvn_mean_kernel)(&arg);

                mean *= C2inv;

                if (normalize_variance) {
                    // variance for this channel
                    float variance = 0.f;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance);
                    (*mvn_variance_kernel)(&arg);

                    if (epsMode_ == insideSqrt)
                        variance = 1.f / sqrtf(variance * C2inv + eps);
                    else if (epsMode_ == outsideSqrt)
                        variance = 1.f / (sqrtf(variance * C2inv) + eps);

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
    std::tie(N, C, D, H, W) = get5dShapes(dims);

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    for (size_t b = 0lu; b < N; b++) {
        size_t cb = b * C3;
        if (across_channels) {
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

            if (normalize_variance) {
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
                if (epsMode_ == insideSqrt)
                    variance = 1.f / sqrtf(variance_temp * C3inv + eps);
                else if (epsMode_ == outsideSqrt)
                    variance = 1.f / (sqrtf(variance_temp * C3inv) + eps);

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

                if (normalize_variance) {
                    // variance for this channel
                    float variance = 0.f;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        variance += (src_data_ptr[cc + sp] - mean) * (src_data_ptr[cc + sp] - mean);
                    }

                    if (epsMode_ == insideSqrt)
                        variance = 1.f / sqrtf(variance * C2inv + eps);
                    else if (epsMode_ == outsideSqrt)
                        variance = 1.f / (sqrtf(variance * C2inv) + eps);

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

void MKLDNNMVNNode::mvn_blk(const std::vector<const uint8_t *>& src_data, uint8_t* dst_data, const SizeVector& dims) {
    size_t input_num = src_data.size();
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(cpu::x64::avx512_common)) {
        blk_size = 16;
    } else {
        blk_size = 8;
    }

    size_t N = 1; size_t C = 1; size_t D = 1; size_t H = 1; size_t W = 1;
    std::tie(N, C, D, H, W) = get5dShapes(dims);

    bool is_nhwc = false;
    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();
    if (layout == NHWC || layout == NDHWC)
        is_nhwc = true;

    size_t CB = div_up(C, blk_size);

    size_t C0 = is_nhwc ? W * C : W * blk_size;
    size_t C1 = C0 * H;
    size_t C2 = C1 * D;
    size_t C3 = C2 * CB;
    size_t C5 = C * D * H * W;

    size_t threads_num = parallel_get_num_threads();
    size_t aux_buffer_size = across_channels ? blk_size : rnd_up(C, blk_size);
    std::vector<float> mean_buffer(aux_buffer_size * threads_num);
    std::vector<float> variance_buffer(aux_buffer_size * threads_num);

    size_t src_stride_size = is_nhwc ? static_cast<size_t>(C * src_data_size) : static_cast<size_t>(blk_size * src_data_size);
    size_t dst_stride_size = is_nhwc ? static_cast<size_t>(C * dst_data_size) : static_cast<size_t>(blk_size * dst_data_size);

    for (size_t b = 0lu; b < N; b++) {
        size_t b_offset = is_nhwc ? b * C5 : b * C3;
        if (across_channels) {
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
                arg.src[0] = src_data[0] + src_offset * src_data_size;
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

            if (normalize_variance) {
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
                    arg.src[0] = src_data[0] + src_offset * src_data_size;
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
                if (epsMode_ == insideSqrt)
                    variance /= sqrtf(variance_temp * C5inv + eps);
                else if (epsMode_ == outsideSqrt)
                    variance /= sqrtf(variance_temp * C5inv) + eps;
                // mvn for one instance in batch
                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                : b_offset + cb * C2 + d * C1 + h * C0;
                    auto arg = jit_mvn_call_args();
                    for (int i = 0; i < input_num; ++i)
                        arg.src[i] = src_data[i] + src_offset * src_data_size;
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
                    for (int i = 0; i < input_num; ++i)
                        arg.src[i] = src_data[i] + src_offset * src_data_size;
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
                    arg.src[0] = src_data[0] + src_offset * src_data_size;
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

            if (normalize_variance) {
                for (int i = 0; i < variance_buffer.size(); i++)
                    variance_buffer[i] = 0.f;

                parallel_for2d(D, H, [&](size_t thr_idx, size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                    : b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                        auto arg = jit_mvn_call_args();
                        arg.src[0] = src_data[0] + src_offset * src_data_size;
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
                    if (epsMode_ == insideSqrt)
                        variance_buffer[c] = 1.f / sqrtf(variance_buffer[c] * size_inv + eps);
                    else if (epsMode_ == outsideSqrt)
                        variance_buffer[c] = 1.f / (sqrtf(variance_buffer[c] * size_inv) + eps);
                }

                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                    : b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb];

                        auto arg = jit_mvn_call_args();
                        for (int i = 0; i < input_num; ++i)
                            arg.src[i] = src_data[i] + src_offset * src_data_size;
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
                // normalize_variance == false
                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = is_nhwc ? b_offset + d * C1 + h * C0 + cb * blk_size
                                                    : b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];

                        auto arg = jit_mvn_call_args();
                        for (int i = 0; i < input_num; ++i)
                            arg.src[i] = src_data[i] + src_offset * src_data_size;
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

// Validates MVN node axes to check whether it can be executed on the current CPU implementation.
// Supported cases:
// 1D: axes: [0]
// 2D: axes: [1]
// 3D: axes: [1,2], [2]
// 4D: axes: [1,2,3], [2,3]
// 5D: axes: [1,2,3,4], [2,3,4]
bool MKLDNNMVNNode::checkAxesSuitability(const std::shared_ptr<const ngraph::Node>& node) {
    const auto mvn = std::dynamic_pointer_cast<const ngraph::op::v6::MVN>(node);
    if (mvn != nullptr && node->get_input_size() == 2) {
        if (auto axesNode = dynamic_cast<ngraph::op::v0::Constant*>(mvn->get_input_node_ptr(1))) {
            auto& mvnShape = mvn->get_output_shape(0);
            auto axesVal = axesNode->cast_vector<int>();
            for (int& axe : axesVal)
                axe = axe < 0 ? axe + mvnShape.size() : axe;
            std::sort(axesVal.begin(), axesVal.end());
            if (mvnShape.size() == 1) {
                if (axesVal.size() == 1 && axesVal[0] == 0)
                    return true;
                else
                    return false;
            }
            if (mvnShape.size() > 5 || (mvnShape.size() != axesVal.size() + 1 && mvnShape.size() != axesVal.size() + 2))
                return false;
            int value = mvnShape.size() - 1;
            for (int i = axesVal.size() - 1; i >= 0; i--, value--) {
                if (axesVal[i] != value)
                    return false;
            }
            return true;
        }
    }
    return false;
}

bool MKLDNNMVNNode::isDepthWiseNode(const MKLDNNNodePtr& node) const {
    InferenceEngine::details::CaselessEq<std::string> comparator;
    auto layerType = node->getCnnLayer().get()->type;
    return node->getType() == Eltwise && (comparator(layerType, "scaleshift") || comparator(layerType, "prelu"));
}

void MKLDNNMVNNode::setHasAxesInput(bool hasAxes) {
    hasAxesInput = hasAxes;
}

bool MKLDNNMVNNode::canFuse(const MKLDNNNodePtr& node) const {
    auto isOneOf = [&](EltwiseOpType alg, std::vector<EltwiseOpType> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    if (!mayiuse(cpu::x64::sse41))
        return false;

    size_t addedInputEdgesNum = (node->getType() == Quantize || isDepthWiseNode(node)) ? 0 : node->getParentEdges().size() - 1;
    int maxMVNAllInput = hasAxesInput ? MAX_MVN_INPUTS + 1 : MAX_MVN_INPUTS;
    if (getParentEdges().size() + addedInputEdgesNum > maxMVNAllInput)
        return false;

    std::string errPrefix = "MVN node with name '" + getName() + "' ";
    if (node->getType() == Quantize) {
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
        if (quantizeNode == nullptr)
            IE_THROW() << errPrefix << "cannot get quantize node to fuse with.";

        return !quantizeNode->isBinarization();
    } else if (node->getType() == Eltwise) {
        if (isDepthWiseNode(node)) {
            return true;
        }
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode*>(node.get());
        if (eltwiseNode == nullptr)
            IE_THROW() << errPrefix << "cannot get eltwise node to fuse with.";

        for (int i = 1; i < eltwiseNode->getCnnLayer()->insData.size(); i++) {
            if (eltwiseNode->getCnnLayer()->insData[0].lock()->getPrecision() != eltwiseNode->getCnnLayer()->insData[i].lock()->getPrecision()) {
                return false;
            }
        }

        if (eltwiseNode->getParentEdgesAtPort(0)[0]->getParent().get() != this) {
            // Eltwise jitter doesn't respect commutative property, so fusing is disabled in case it applied not for 0-th port.
            if (isOneOf(eltwiseNode->getOpType(), {Subtract, Divide, FloorMod, Mod, PowerDynamic, Greater, GreaterEqual, Less, LessEqual})) {
                return false;
            }
        }
        return true;
    }

    return false;
}

bool MKLDNNMVNNode::created() const {
    return getType() == MVN;
}

REG_MKLDNN_PRIM_FOR(MKLDNNMVNNode, MVN);
