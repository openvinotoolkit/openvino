// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <algorithm>
#include <ie_parallel.hpp>
#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "softmax.h"

using namespace InferenceEngine;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;

#define GET_OFF(field) offsetof(jit_args_softmax, field)

struct jit_args_softmax {
    const float* src;
    const float* dst;
    size_t stride;
    size_t work_amount;
};

struct jit_uni_softmax_kernel {
    void (*ker_)(const jit_args_softmax *);

    void operator()(const jit_args_softmax *args) { assert(ker_); ker_(args); }

    jit_uni_softmax_kernel() : ker_(nullptr) {}
    virtual ~jit_uni_softmax_kernel() {}
};

template <cpu_isa_t isa>
struct jit_uni_softmax_kernel_f32 : public jit_uni_softmax_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_softmax_kernel_f32)

    jit_uni_softmax_kernel_f32() : jit_uni_softmax_kernel(), jit_generator() {
        exp_injector.reset(new jit_uni_eltwise_injector_f32<isa>(this, alg_kind::eltwise_exp, 0.f, 0.f));

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_stride, ptr[reg_params + GET_OFF(stride)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        Xbyak::Label max_loop_label;
        Xbyak::Label max_loop_end_label;
        Xbyak::Label exp_loop_label;
        Xbyak::Label exp_loop_end_label;
        Xbyak::Label div_loop_label;
        Xbyak::Label div_loop_end_label;

        mov(aux_reg_work_amount, reg_work_amount);
        mov(aux_reg_src, reg_src);
        uni_vmovups(vmm_max, ptr[aux_reg_src]);
        L(max_loop_label); {
            cmp(aux_reg_work_amount, 0);
            jle(max_loop_end_label, T_NEAR);

            uni_vmovups(vmm_val, ptr[aux_reg_src]);

            if (isa == sse42) {
                uni_vmovups(vmm_mask, vmm_val);
                uni_vcmpgtps(vmm_mask, vmm_mask, vmm_max);
            } else if (isa == avx2) {
                uni_vcmpgtps(vmm_mask, vmm_val, vmm_max);
            } else {
                vcmpps(k_mask, vmm_val, vmm_max, _cmp_nle_us);
            }

            if (isa == avx512_common) {
                vptestmd(k_mask, vmm_mask, vmm_mask);
                vblendmps(vmm_max | k_mask, vmm_max, vmm_val);
            } else {
                uni_vblendvps(vmm_max, vmm_max, vmm_val, vmm_mask);
            }

            add(aux_reg_src, reg_stride);
            sub(aux_reg_work_amount, 1);

            jmp(max_loop_label, T_NEAR);
        }

        L(max_loop_end_label);

        mov(aux_reg_work_amount, reg_work_amount);
        mov(aux_reg_src, reg_src);
        mov(aux_reg_dst, reg_dst);
        uni_vpxor(vmm_exp_sum, vmm_exp_sum, vmm_exp_sum);
        L(exp_loop_label); {
            cmp(aux_reg_work_amount, 0);
            jle(exp_loop_end_label, T_NEAR);

            uni_vmovups(vmm_val, ptr[aux_reg_src]);

            uni_vsubps(vmm_val, vmm_val, vmm_max);
            exp_injector->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
            uni_vaddps(vmm_exp_sum, vmm_exp_sum, vmm_val);

            uni_vmovups(ptr[aux_reg_dst], vmm_val);

            add(aux_reg_src, reg_stride);
            add(aux_reg_dst, reg_stride);
            sub(aux_reg_work_amount, 1);

            jmp(exp_loop_label, T_NEAR);
        }

        L(exp_loop_end_label);

        mov(aux_reg_work_amount, reg_work_amount);
        mov(aux_reg_dst, reg_dst);
        L(div_loop_label); {
            cmp(aux_reg_work_amount, 0);
            jle(div_loop_end_label, T_NEAR);

            uni_vmovups(vmm_val, ptr[aux_reg_dst]);

            uni_vdivps(vmm_val, vmm_val, vmm_exp_sum);

            uni_vmovups(ptr[aux_reg_dst], vmm_val);

            add(aux_reg_dst, reg_stride);
            sub(aux_reg_work_amount, 1);

            jmp(div_loop_label, T_NEAR);
        }

        L(div_loop_end_label);

        this->postamble();

        exp_injector->prepare_table();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == sse42, Xbyak::Xmm, isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 aux_reg_src = r13;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 aux_reg_dst = r15;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 aux_reg_work_amount = r12;
    Xbyak::Reg64 reg_stride = r14;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_mask = Vmm(0);
    Vmm vmm_val = Vmm(1);
    Vmm vmm_max = Vmm(2);
    Vmm vmm_exp_sum = Vmm(3);

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);

    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector;
};

SoftmaxGeneric::SoftmaxGeneric() {
    block_size = 1;
    if (mayiuse(avx512_common)) {
        softmax_kernel.reset(new jit_uni_softmax_kernel_f32<avx512_common>());
        block_size = 16;
    } else if (mayiuse(avx2)) {
        softmax_kernel.reset(new jit_uni_softmax_kernel_f32<avx2>());
        block_size = 8;
    } else if (mayiuse(sse42)) {
        softmax_kernel.reset(new jit_uni_softmax_kernel_f32<sse42>());
        block_size = 4;
    }
}

void SoftmaxGeneric::execute(const float *src_data, float *dst_data, int B, int C, int H, int W) {
    for (int b = 0; b < B; b++) {
        int tail_start = 0;
        if (softmax_kernel) {
            int blocks_num = H*W / block_size;

            parallel_for(blocks_num, [&](int ib) {
                auto arg = jit_args_softmax();

                arg.src = src_data + b * C * H * W + ib * block_size;
                arg.dst = dst_data + b * C * H * W + ib * block_size;
                arg.stride = static_cast<size_t>((size_t)(H) * W * sizeof(float));
                arg.work_amount = static_cast<size_t>(C);

                (*softmax_kernel)(&arg);
            });

            tail_start = (H*W / block_size) * block_size;
        }

        parallel_for(H * W - tail_start, [&](int i) {
            int offset = i + tail_start;
            float max = src_data[b * C * H * W + offset];
            for (int c = 0; c < C; c++) {
                float val = src_data[b * C * H * W + c * H * W + offset];
                if (val > max) max = val;
            }

            float expSum = 0;
            for (int c = 0; c < C; c++) {
                dst_data[b * C * H * W + c * H * W + offset] = exp(src_data[b * C * H * W + c * H * W + offset] - max);
                expSum += dst_data[b * C * H * W + c * H * W + offset];
            }

            for (int c = 0; c < C; c++) {
                dst_data[b * C * H * W + c * H * W + offset] = dst_data[b * C * H * W + c * H * W + offset] / expSum;
            }
        });
    }
}
