// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_quantize_node.h"
#include "mkldnn_depthwise_node.h"
#include "mkldnn_activation_node.h"
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>
#include "ie_parallel.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"
#include "jit_uni_quantization.hpp"
#include "bf16transformer.h"

#include "mkldnn_normalize_node.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_normalize_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_normalize_modulo_kernel_f32 : public jit_uni_normalize_modulo_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_normalize_modulo_kernel_f32)

    jit_uni_normalize_modulo_kernel_f32(jit_normalize_config_params jcp) : jit_uni_normalize_modulo_kernel(jcp), jit_generator() {
        this->preamble();
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_modulo, ptr[reg_params + GET_OFF(modulo)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_src_stride, ptr[reg_params + GET_OFF(src_stride)]);

        Xbyak::Label modulo_loop_label;
        Xbyak::Label modulo_loop_end_label;

        uni_vpxor(vmm_sqr_sum, vmm_sqr_sum, vmm_sqr_sum);
        L(modulo_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(modulo_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            uni_vfmadd231ps(vmm_sqr_sum, vmm_val, vmm_val);
            if (isa == cpu::sse42 && jcp_.is_blk) {
                int sse42_offset = 4;
                load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                uni_vfmadd231ps(vmm_sqr_sum, vmm_val, vmm_val);
            }

            add(reg_src, reg_src_stride);
            sub(reg_work_amount, 1);

            jmp(modulo_loop_label, T_NEAR);
        }
        L(modulo_loop_end_label);

        if (jcp_.is_nchw && !jcp_.across_spatial) {
            uni_vmovups(ptr[reg_modulo], vmm_sqr_sum);
        } else {
            // hsum+store
            if (isa == cpu::sse42) {
                hsum_store(vmm_sqr_sum);
            } else if (isa == cpu::avx2) {
                Xbyak::Ymm ymm_sqr_sum = Xbyak::Ymm(vmm_sqr_sum.getIdx());
                vextractf128(xmm_aux1, ymm_sqr_sum, 0);
                vextractf128(xmm_aux2, ymm_sqr_sum, 1);
                addps(xmm_aux1, xmm_aux2);
                hsum_store(xmm_aux1);
            } else {
                Xbyak::Zmm zmm_sqr_sum = Xbyak::Zmm(vmm_sqr_sum.getIdx());
                vextractf32x4(xmm_aux1, zmm_sqr_sum, 0);
                vextractf32x4(xmm_aux2, zmm_sqr_sum, 1);
                addps(xmm_aux1, xmm_aux2);
                vextractf32x4(xmm_aux2, zmm_sqr_sum, 2);
                vextractf32x4(xmm_aux3, zmm_sqr_sum, 3);
                addps(xmm_aux2, xmm_aux3);
                addps(xmm_aux1, xmm_aux2);
                hsum_store(xmm_aux1);
            }
        }

        this->postamble();
        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_work_amount = r9;
    Xbyak::Reg64 reg_src_stride = r10;
    Xbyak::Reg64 reg_modulo = rbp;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_sqr_sum = Vmm(1);
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(2);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(3);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(4);

    inline void hsum_store(Xbyak::Xmm xmm_sqr_sum) {
        movshdup(xmm_aux3, xmm_sqr_sum);  //  sqrt_sum:1,2,3,4; aux3:2,2,4,4
        addps(xmm_sqr_sum, xmm_aux3);     //  sqrt_sum:1+2,2+2,3+4,4+4
        movhlps(xmm_aux3, xmm_sqr_sum);   //  aux3:3+4,4+4,4,4
        addps(xmm_sqr_sum, xmm_aux3);     //  sqrt_sum:1+2+3+4,...
        movss(ptr[reg_modulo], xmm_sqr_sum);
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
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != memory::f32)
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }
};

// dst = src * modulo_inv * scale
template <cpu_isa_t isa>
struct jit_uni_normalize_kernel_f32 : public jit_uni_normalize_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_normalize_kernel_f32)

    explicit jit_uni_normalize_kernel_f32(jit_normalize_config_params jcp, const mkldnn_primitive_attr &attr)
    : jit_uni_normalize_kernel(jcp, attr), jit_generator() {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this, post_op.depthwise.alg));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_modulo, ptr[reg_params + GET_OFF(modulo)]);
        mov(reg_weights, ptr[reg_params + GET_OFF(weights)]);
        mov(reg_fused_factor, ptr[reg_params + GET_OFF(fused_factor)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        if (attr_.post_ops_.len_ != 0)
            mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);
        if (isa == avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        if (jcp_.is_nchw) {
            normalize_nchw();
        } else if (jcp_.is_blk) {
            normalize_blk();
        } else if (jcp_.is_nhwc) {
            normalize_nhwc();
        }

        this->postamble();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();

        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_modulo = r10;
    Xbyak::Reg64 reg_weights = r11;
    Xbyak::Reg64 reg_fused_factor = r12;
    Xbyak::Reg64 reg_work_amount = r15;
    Xbyak::Reg64 reg_params = abi_param1;

    Reg8 reg_tmp_8 = r14b;
    Reg32 reg_tmp_32 = r14d;
    Reg64 reg_tmp_64 = r14;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = rdx;

    Vmm vmm_val = Vmm(0);
    Xmm xmm_val = Xmm(0);
    Vmm vmm_scale = Vmm(1);
    Xmm xmm_scale = Xmm(1);
    Vmm vmm_modulo = Vmm(2);
    Xmm xmm_modulo = Xmm(2);
    Vmm vmm_fused_factor = Vmm(3);
    Xmm xmm_fused_factor = Xmm(3);
    Vmm vmm_fused_factor2 = Vmm(4);
    Xmm xmm_fused_factor2 = Xmm(4);

    Vmm vmm_d_weights = Vmm(5);
    Vmm vmm_d_bias = Vmm(6);
    Vmm vmm_zero = Vmm(7);

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    inline void normalize_nchw() {
        if (jcp_.across_spatial) {
            uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);  // for channel_shared: false or true.
        } else {
            if (!jcp_.channel_shared) {
                uni_vbroadcastss(vmm_scale, ptr[reg_weights]);
            }
        }

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        int step = vlen / sizeof(float);
        L(main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(main_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            if (jcp_.across_spatial) {
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
            } else {
                if (jcp_.channel_shared) {
                    uni_vmovups(vmm_fused_factor, ptr[reg_fused_factor]);
                    uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
                    add(reg_fused_factor, vlen);
                } else {
                    uni_vmovups(vmm_modulo, ptr[reg_modulo]);  // modulo: ld dynamic
                    uni_vmulps(vmm_val, vmm_val, vmm_modulo);
                    uni_vmulps(vmm_val, vmm_val, vmm_scale);    // weight: bc once
                    add(reg_modulo, vlen);
                }
            }
            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, 1);
            }
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
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

            load_scalar(xmm_val, ptr[reg_src], jcp_.src_dt);
            if (jcp_.across_spatial) {
                uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);
            } else {
                if (jcp_.channel_shared) {
                    load_scalar(xmm_fused_factor, ptr[reg_fused_factor], memory::f32);
                    uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);
                    add(reg_fused_factor, step * sizeof(float));
                } else {
                    load_scalar(xmm_modulo, ptr[reg_modulo], memory::f32);
                    uni_vmulps(xmm_val, xmm_val, xmm_modulo);
                    uni_vmulps(xmm_val, xmm_val, xmm_scale);
                    add(reg_modulo, step * sizeof(float));
                }
            }
            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, 1);  // vector and boradcast
            }
            store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    inline void normalize_nhwc() {
        if (jcp_.channel_shared) {
            uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);
        } else {
            if (!jcp_.across_spatial) {
                uni_vbroadcastss(vmm_modulo, ptr[reg_modulo]);
            }
        }

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        int step = vlen / sizeof(float);
        L(main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(main_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            if (jcp_.channel_shared) {
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
            } else {
                if (jcp_.across_spatial) {
                    uni_vmovups(vmm_fused_factor, ptr[reg_fused_factor]);
                    uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
                    add(reg_fused_factor, vlen);
                } else {
                    uni_vmovups(vmm_scale, ptr[reg_weights]);
                    uni_vmulps(vmm_val, vmm_val, vmm_scale);
                    uni_vmulps(vmm_val, vmm_val, vmm_modulo);
                    add(reg_weights, vlen);
                }
            }
            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, 0);
                add(reg_oc_off, vlen);  // out channel offset of fused ops weights in byte
            }
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
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

            load_scalar(xmm_val, ptr[reg_src], jcp_.src_dt);
            if (jcp_.channel_shared) {
                uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);
            } else {
                if (jcp_.across_spatial) {
                    load_scalar(xmm_fused_factor, ptr[reg_fused_factor], memory::f32);
                    uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);
                    add(reg_fused_factor, step * sizeof(float));
                } else {
                    load_scalar(xmm_scale, ptr[reg_weights], memory::f32);
                    uni_vmulps(xmm_val, xmm_val, xmm_scale);
                    uni_vmulps(xmm_val, xmm_val, xmm_modulo);
                    add(reg_weights, step * sizeof(float));
                }
            }
            if (attr_.post_ops_.len_ != 0) {
                apply_post_ops(jcp_.dst_dt, 0);
                add(reg_oc_off, step * sizeof(float));
            }
            store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

// tails with padding as a vector for normalize.
    inline void normalize_blk() {
        size_t blk_size = 0;
        size_t simd_w = 0;
        if (isa == cpu::avx512_common) {
            blk_size = simd_w = 16;
        } else if (isa == cpu::avx2) {
            blk_size = simd_w = 8;
        } else {
            blk_size = 8;
            simd_w = 4;
        }
        bool is_sse42 = (isa == cpu::sse42);

        if (jcp_.across_spatial) {
            if (jcp_.channel_shared) {
                uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);
            } else {
                uni_vmovups(vmm_fused_factor, ptr[reg_fused_factor]);
                if (is_sse42) {
                    uni_vmovups(vmm_fused_factor2, ptr[reg_fused_factor + simd_w * sizeof(float)]);
                }
            }

            Xbyak::Label norm_loop_label;
            Xbyak::Label norm_loop_end_label;

            L(norm_loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(norm_loop_end_label, T_NEAR);

                load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);

                if (attr_.post_ops_.len_ != 0) {
                    apply_post_ops(jcp_.dst_dt, 0);
                }
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                if (is_sse42) {
                    int sse42_offset = 4;
                    load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                    if (jcp_.channel_shared) {
                        uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);  // bc once
                    } else {
                        uni_vmulps(vmm_val, vmm_val, vmm_fused_factor2);  // ld once
                    }
                    if (attr_.post_ops_.len_ != 0) {
                        add(reg_oc_off, sse42_offset * sizeof(float));
                        apply_post_ops(jcp_.dst_dt, 0);
                        sub(reg_oc_off, sse42_offset * sizeof(float));
                    }
                    store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
                }
                add(reg_src, blk_size * jcp_.src_data_size);
                add(reg_dst, blk_size * jcp_.dst_data_size);

                sub(reg_work_amount, 1);
                jmp(norm_loop_label, T_NEAR);
            }
            L(norm_loop_end_label);
        } else {  // across_saptail is flase
            if (jcp_.channel_shared) {
                uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);
            } else {
                uni_vbroadcastss(vmm_modulo, ptr[reg_modulo]);
            }
            size_t src_stride = jcp_.w * jcp_.h * blk_size * jcp_.src_data_size;
            size_t dst_stride = jcp_.w * jcp_.h * blk_size * jcp_.dst_data_size;

            Xbyak::Label norm_loop_label;
            Xbyak::Label norm_loop_end_label;

            L(norm_loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(norm_loop_end_label, T_NEAR);

                load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
                if (jcp_.channel_shared) {
                    uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
                } else {
                    uni_vmovups(vmm_scale, ptr[reg_weights]);
                    uni_vmulps(vmm_val, vmm_val, vmm_scale);
                    uni_vmulps(vmm_val, vmm_val, vmm_modulo);
                    add(reg_weights, vlen);
                }
                if (attr_.post_ops_.len_ != 0) {
                    apply_post_ops(jcp_.dst_dt, 0);
                    add(reg_oc_off, vlen);  // vlen is related isa
                }
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                if (is_sse42) {
                    int sse42_offset = 4;
                    load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                    if (jcp_.channel_shared) {
                        uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);  // bc once
                    } else {
                        uni_vmovups(vmm_scale, ptr[reg_weights]);  // ld dynamic
                        uni_vmulps(vmm_val, vmm_val, vmm_scale);
                        uni_vmulps(vmm_val, vmm_val, vmm_modulo);  // bc once
                        add(reg_weights, vlen);  // 4 * sizeof(float)
                    }
                    if (attr_.post_ops_.len_ != 0) {
                        apply_post_ops(jcp_.dst_dt, 0);
                        add(reg_oc_off, vlen);  // vlen is related isa
                    }
                    store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
                }
                add(reg_src, src_stride);
                add(reg_dst, dst_stride);

                sub(reg_work_amount, 1);
                jmp(norm_loop_label, T_NEAR);
            }
            L(norm_loop_end_label);
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
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != memory::f32)
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
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != data_type::f32) {
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
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt) {
        if (dst_dt != data_type::f32) {
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
            default:
                assert(!"unknown dst_dt");
        }
    }

    // scalar: load scalar to xmm, process on xmm with padded param, store xmm to scalar.
    // is_broadcast for broadcasting param for depth_wise and quantize, for fusion with plain layout.
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
                // weight and bias is padding. scalar as vector.
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

MKLDNNNormalizeNode::MKLDNNNormalizeNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache), src_data_size(0lu), dst_data_size(0lu), weights_data_size(0lu),
        input_prec(Precision::UNSPECIFIED), output_prec(Precision::UNSPECIFIED), weights_prec(Precision::UNSPECIFIED) {}

void MKLDNNNormalizeNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    if (getParentEdgeAt(0)->getDims().ndims() > 4 || getParentEdgeAt(0)->getDims().ndims() < 2) {
        THROW_IE_EXCEPTION << "Normalize supports from 2D to 4D blobs!";
    }

    auto *layer = getCnnLayer().get();
    if (layer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get Normalize layer.";
    across_spatial = layer->GetParamAsBool("across_spatial", false);
    channel_shared = layer->GetParamAsBool("channel_shared", false);
    eps = layer->GetParamAsFloat("eps");

    MemoryBlob::Ptr tweights = as<MemoryBlob>(layer->blobs.at("weights"));
    if (!tweights) {
        THROW_IE_EXCEPTION << layer->name << "Weights are not initialized or cannot be casted to MemoryBlob for layer Normalize with name '"
            << layer->name << "'";
    }
    weights_prec = tweights->getTensorDesc().getPrecision();

    if (weights_prec == Precision::FP32) {
        weights_blob = tweights;
    } else if (weights_prec == Precision::BF16) {
        MKLDNNPlugin::BF16Transformer transformer;
        weights_blob = transformer.convertBF16ToFloat(tweights);
    } else {
        // Unknown non supported data type, return an error
        THROW_IE_EXCEPTION << layer->name << "Weights for layer Normalize with name '" << layer->name <<
            "' has unsupported data type " << tweights->getTensorDesc().getPrecision();
    }
}

void MKLDNNNormalizeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    setPostOps(attr, true);

    Precision inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    inputPrecision = inputPrecision == Precision::BF16 ? Precision(Precision::FP32) : inputPrecision;
    Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();
    outputPrecision = outputPrecision == Precision::BF16 ? Precision(Precision::FP32) : outputPrecision;

    if (!fusedWith.empty()) {
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputPrecision = lastFusedLayer->outData[0]->getPrecision();
        }
    }

    auto isOneOf = [&](InferenceEngine::Precision precision, std::vector<InferenceEngine::Precision> precisions) {
        for (auto p : precisions) {
            if (precision == p) {
                return true;
            }
        }
        return false;
    };
    if (!isOneOf(inputPrecision, {Precision::FP32, Precision::I8, Precision::U8})) {
        THROW_IE_EXCEPTION << "Unsupported input precision. " << getName();
    }
    if (!isOneOf(outputPrecision, {Precision::FP32, Precision::I8, Precision::U8})) {
        THROW_IE_EXCEPTION << "Unsupported output precision. " << getName();
    }
    if (!isOneOf(weights_prec, {Precision::FP32, Precision::BF16})) {
        THROW_IE_EXCEPTION << "Unsupported wights precision. " << getName();
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);
    auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(weights_prec);

    input_prec = inputPrecision;
    output_prec = outputPrecision;
    src_data_size = MKLDNNExtensionUtils::sizeOfDataType(inputDataType);
    dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(outputDataType);
    weights_data_size = MKLDNNExtensionUtils::sizeOfDataType(weightsDataType);

    bool canBeInplace = src_data_size == dst_data_size && getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].constant = false;
    config.outConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.outConfs[0].inPlace = canBeInplace ? 0 : -1;

    auto pushDesc = [&](memory::format format) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, format);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), outputDataType, format);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
    };

    // only plain layout support when w/o sse42
    if (getParentEdgeAt(0)->getDims().ndims() == 4) {
        if (mayiuse(cpu::sse42)) {
            pushDesc(memory::nhwc);
            if (mayiuse(cpu::avx512_common)) {
                pushDesc(memory::nChw16c);
            } else {
                pushDesc(memory::nChw8c);
            }
        }
    }
    if (canBeInplace)
        config.inConfs[0].inPlace = 0;
    pushDesc(MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims()));
}

void MKLDNNNormalizeNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
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
                MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(getParentEdgeAt(0)->getDims()[1], 16))});

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
                } else {
                    ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         nullptr);

                    blob_idx += 1;
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

void MKLDNNNormalizeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    auto selectedPD = getSelectedPrimitiveDescriptor();
    Layout selected_layout = selectedPD->getConfig().inConfs[0].desc.getLayout();
    auto jcp = jit_normalize_config_params();
    jcp.src_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[0].desc.getPrecision());
    jcp.dst_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().outConfs[0].desc.getPrecision());
    jcp.src_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.src_dt);
    jcp.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.dst_dt);
    jcp.is_nchw = selected_layout == MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims());
    jcp.is_nhwc = selected_layout == Layout::NHWC;
    jcp.is_blk = selected_layout == Layout::BLOCKED;
    jcp.across_spatial = across_spatial;
    jcp.channel_shared = channel_shared;
    auto dims = getParentEdgeAt(0)->getDesc().getDims();
    size_t dims_size = dims.size();
    jcp.n = (dims_size > 0) ? dims[0] : 1lu;
    jcp.c = (dims_size > 1) ? dims[1] : 1lu;
    jcp.h = (dims_size > 2) ? dims[2] : 1lu;
    jcp.w = (dims_size > 3) ? dims[3] : 1lu;

    if (mayiuse(cpu::avx512_common)) {
        normalize_modulo_kernel.reset(new jit_uni_normalize_modulo_kernel_f32<cpu::avx512_common>(jcp));
        normalize_kernel.reset(new jit_uni_normalize_kernel_f32<cpu::avx512_common>(jcp, *attr.get()));
    } else if (mayiuse(cpu::avx2)) {
        normalize_modulo_kernel.reset(new jit_uni_normalize_modulo_kernel_f32<cpu::avx2>(jcp));
        normalize_kernel.reset(new jit_uni_normalize_kernel_f32<cpu::avx2>(jcp, *attr.get()));
    } else if (mayiuse(cpu::sse42)) {
        normalize_modulo_kernel.reset(new jit_uni_normalize_modulo_kernel_f32<cpu::sse42>(jcp));
        normalize_kernel.reset(new jit_uni_normalize_kernel_f32<cpu::sse42>(jcp, *attr.get()));
    }

    const auto &p = (*attr.get()).post_ops_;
    for (int i = 0; i < p.len_; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors_ref.push_back(std::make_shared<ref_eltwise_scalar_fwd_t>(
                post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors_ref.push_back(std::make_shared<ref_depthwise_scalar_fwd_t>(
                    post_op.depthwise.alg));
        }
    }
}

void MKLDNNNormalizeNode::execute(mkldnn::stream strm) {
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    const uint8_t *src_ptr = reinterpret_cast<const uint8_t*>(srcMemPtr->GetData()) +
            srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding *
            MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(srcMemPtr->GetDescriptor().data.data_type));
    uint8_t *dst_ptr = reinterpret_cast<uint8_t*>(dstMemPtr->GetData()) +
            dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding *
            MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dstMemPtr->GetDescriptor().data.data_type));

    auto dims = getParentEdgeAt(0)->getDesc().getDims();

    if (output_prec == Precision::U8) {
        auto dst_data = reinterpret_cast<uint8_t *>(dst_ptr);
        if (input_prec == Precision::U8) {
            auto src_data = reinterpret_cast<const uint8_t *>(src_ptr);
            normalize_function<uint8_t, uint8_t>(src_data, dst_data, dims);
        } else if (input_prec == Precision::I8) {
            auto src_data = reinterpret_cast<const int8_t *>(src_ptr);
            normalize_function<int8_t, uint8_t>(src_data, dst_data, dims);
        } else if (input_prec == Precision::FP32) {
            auto src_data = reinterpret_cast<const float *>(src_ptr);
            normalize_function<float, uint8_t>(src_data, dst_data, dims);
        }
    } else if (output_prec == Precision::I8) {
        auto dst_data = reinterpret_cast<int8_t *>(dst_ptr);
        if (input_prec == Precision::U8) {
            auto src_data = reinterpret_cast<const uint8_t *>(src_ptr);
            normalize_function<uint8_t, int8_t>(src_data, dst_data, dims);
        } else if (input_prec == Precision::I8) {
            auto src_data = reinterpret_cast<const int8_t *>(src_ptr);
            normalize_function<int8_t, int8_t>(src_data, dst_data, dims);
        } else if (input_prec == Precision::FP32) {
            auto src_data = reinterpret_cast<const float *>(src_ptr);
            normalize_function<float, int8_t>(src_data, dst_data, dims);
        }
    } else if (output_prec == Precision::FP32) {
        auto dst_data = reinterpret_cast<float *>(dst_ptr);
        if (input_prec == Precision::U8) {
            auto src_data = reinterpret_cast<const uint8_t *>(src_ptr);
            normalize_function<uint8_t, float>(src_data, dst_data, dims);
        } else if (input_prec == Precision::I8) {
            auto src_data = reinterpret_cast<const int8_t *>(src_ptr);
            normalize_function<int8_t, float>(src_data, dst_data, dims);
        } else if (input_prec == Precision::FP32) {
            auto src_data = reinterpret_cast<const float *>(src_ptr);
            normalize_function<float, float>(src_data, dst_data, dims);
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeNode::normalize_nchw(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims) {
    size_t blk_size = 1;  // elt in vmm
    if (mayiuse(cpu::avx512_common)) {
        blk_size = 16;
    } else if (mayiuse(cpu::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::sse42)) {
        blk_size = 4;
    }

    size_t dims_size = dims.size();
    size_t W = (dims_size > 3) ? dims[3] : 1lu;
    size_t H = (dims_size > 2) ? dims[2] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t B = (dims_size > 0) ? dims[0] : 1lu;
    float *weights = weights_blob->buffer().as<float *>();

    for (size_t b = 0lu; b < B; b++) {
        const in_data_t *src_data_b = src_data + b * C * H * W;
        out_data_t *dst_data_b = dst_data + b * C * H * W;
        if (across_spatial) {
            // modulo
            float addition_identity = 0.0f;
            float modulo = 0.0f;
            modulo = parallel_sum(C, addition_identity, [&](int ic) -> float {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                float modulo_kernel = 0.0f;
                float modulo_tail = 0.0f;
                size_t tail_start = 0;

                auto arg = jit_normalize_call_args();
                arg.src = src_data_bc;
                arg.modulo = static_cast<float*>(&modulo_kernel);
                arg.src_stride = blk_size * sizeof(in_data_t);
                arg.work_amount = (W * H) / blk_size;
                (*normalize_modulo_kernel)(&arg);

                tail_start = (W * H / blk_size) * blk_size;

                // tail
                for (size_t tail = tail_start; tail < H * W; tail++) {
                    modulo_tail += src_data_bc[tail] * src_data_bc[tail];
                }
                return modulo_kernel + modulo_tail;
            });

            modulo = std::sqrt(modulo);
            float modulo_inv = 1.0f / (modulo + eps);

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                out_data_t *dst_data_bc = dst_data_b + ic * H * W;
                float fused_weight_modulo = channel_shared ? (weights[0] * modulo_inv) : (weights[ic] * modulo_inv);
                auto arg = jit_normalize_call_args();
                arg.src = src_data_bc;
                arg.dst = dst_data_bc;
                arg.fused_factor = static_cast<float*>(&fused_weight_modulo);  // broadcast once
                arg.oc_off = ic * sizeof(float);
                arg.work_amount = static_cast<size_t>(W * H);
                (*normalize_kernel)(&arg);
            });
        } else {  // across_spatial: false
            // moduloM
            std::vector<float> moduloM(H * W, 0.f);
            size_t blocks_num = div_up(H * W, blk_size);
            parallel_for(blocks_num, [&](size_t ib) {
                const in_data_t *src_data_b_ib = src_data_b + ib * blk_size;
                size_t min_cb = (std::min)(blk_size, (H * W) - (ib * blk_size));
                if (min_cb == blk_size) {
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_b_ib;
                    arg.modulo = static_cast<float*>(&moduloM[ib * blk_size]);
                    arg.src_stride = W * H * sizeof(in_data_t);
                    arg.work_amount = C;
                    (*normalize_modulo_kernel)(&arg);
                } else {
                    for (size_t c = 0; c < C; c++) {
                        const in_data_t *src_data_b_ib_c = src_data_b_ib + W * H * c;
                        for (size_t blk = 0; blk < min_cb; blk++) {
                            moduloM[ib * blk_size + blk] += src_data_b_ib_c[blk] * src_data_b_ib_c[blk];
                        }
                    }
                }
            });

            for (size_t m = 0; m < H * W; m++) {
                moduloM[m] = 1.0f / (std::sqrt(moduloM[m]) + eps);
                if (channel_shared)
                    moduloM[m] = moduloM[m] * weights[0];
            }

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                out_data_t *dst_data_bc = dst_data_b + ic * H * W;
                auto arg = jit_normalize_call_args();
                arg.src = src_data_bc;
                arg.dst = dst_data_bc;
                if (channel_shared) {
                    arg.fused_factor = static_cast<float*>(&moduloM[0]);  // ld dynamic
                } else {
                    arg.modulo = static_cast<float*>(&moduloM[0]);    // ld dynamic
                    arg.weights = static_cast<float*>(&weights[ic]);  // bc once
                }
                arg.oc_off = ic * sizeof(float);
                arg.work_amount = static_cast<size_t>(W * H);
                (*normalize_kernel)(&arg);
            });
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeNode::normalize_nchw_ref(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims) {
    size_t dims_size = dims.size();
    size_t W = (dims_size > 3) ? dims[3] : 1lu;
    size_t H = (dims_size > 2) ? dims[2] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t B = (dims_size > 0) ? dims[0] : 1lu;
    float *weights = weights_blob->buffer().as<float *>();

    for (size_t b = 0lu; b < B; b++) {
        const in_data_t *src_data_b = src_data + b * C * H * W;
        out_data_t *dst_data_b = dst_data + b * C * H * W;
        if (across_spatial) {
            // modulo
            float addition_identity = 0.0f;
            float modulo = 0.0f;
            modulo = parallel_sum(C, addition_identity, [&](int ic) -> float {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                float modulo_c = 0.0f;
                for (size_t m = 0; m < H * W; m++) {
                    modulo_c += src_data_bc[m] * src_data_bc[m];
                }
                return modulo_c;
            });

            modulo = std::sqrt(modulo);
            float modulo_inv = 1.0f / (modulo + eps);

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                out_data_t *dst_data_bc = dst_data_b + ic * H * W;
                float fused_weight_modulo = channel_shared ? (weights[0] * modulo_inv) : (weights[ic] * modulo_inv);
                for (size_t m = 0; m < W * H; m++) {
                    float dst_value = src_data_bc[m] * fused_weight_modulo;
                    apply_post_ops_scalar(dst_value, ic);
                    if (output_prec == Precision::U8) {
                        dst_data_bc[m] = (dst_value >= 0) ? dst_value : 0;
                    } else {
                        dst_data_bc[m] = dst_value;
                    }
                }
            });
        } else {  // across_spatial: false
            // moduloM
            std::vector<float> moduloM(H * W, 0.f);
            parallel_for(H, [&](size_t ih) {
                size_t offset_h = ih * W;
                const in_data_t *src_data_b_ih = src_data_b + offset_h;
                for (size_t c = 0; c < C; c++) {
                    const in_data_t *src_data_b_ih_c = src_data_b_ih + W * H * c;
                    for (size_t w = 0; w < W; w++) {
                        moduloM[offset_h + w] += src_data_b_ih_c[w] * src_data_b_ih_c[w];
                    }
                }
            });

            for (size_t m = 0; m < H * W; m++) {
                moduloM[m] = 1.0f / (std::sqrt(moduloM[m]) + eps);
                if (channel_shared)
                    moduloM[m] = moduloM[m] * weights[0];
            }

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                out_data_t *dst_data_bc = dst_data_b + ic * H * W;
                for (size_t m = 0; m < W * H; m++) {
                    float dst_value = channel_shared ? src_data_bc[m] * moduloM[m] :
                                      src_data_bc[m] * moduloM[m] * weights[ic];
                    apply_post_ops_scalar(dst_value, ic);
                    if (output_prec == Precision::U8) {
                        dst_data_bc[m] = (dst_value >= 0) ? dst_value : 0;
                    } else {
                        dst_data_bc[m] = dst_value;
                    }
                }
            });
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeNode::normalize_nhwc(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims) {
    size_t blk_size = 1;  // elt in vmm
    if (mayiuse(cpu::avx512_common)) {
        blk_size = 16;
    } else if (mayiuse(cpu::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::sse42)) {
        blk_size = 4;
    }

    size_t dims_size = dims.size();
    size_t W = (dims_size > 3) ? dims[3] : 1lu;
    size_t H = (dims_size > 2) ? dims[2] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t B = (dims_size > 0) ? dims[0] : 1lu;
    float *weights = weights_blob->buffer().as<float *>();

    for (size_t b = 0lu; b < B; b++) {
        const in_data_t *src_data_b = src_data + b * C * H * W;
        out_data_t *dst_data_b = dst_data + b * C * H * W;
        if (across_spatial) {
            // modulo
            float addition_identity = 0;
            float modulo = 0.0f;
            modulo = parallel_sum(H, addition_identity, [&](int ih) -> float {
                size_t tail_start = 0;
                const in_data_t *src_data_bh = src_data_b + ih * C * W;
                float modulo_kernel = 0.f;
                float modulo_tail = 0.f;

                auto arg = jit_normalize_call_args();
                arg.src = src_data_bh;
                arg.modulo = static_cast<float*>(&modulo_kernel);
                arg.src_stride = blk_size * sizeof(in_data_t);
                arg.work_amount = (C * W) / blk_size;
                (*normalize_modulo_kernel)(&arg);

                tail_start = (C * W / blk_size) * blk_size;

                // tail
                for (size_t tail = tail_start; tail < C * W; tail++) {
                    modulo_tail += src_data_bh[tail] * src_data_bh[tail];
                }
                return modulo_kernel + modulo_tail;
            });
            modulo = std::sqrt(modulo);
            float modulo_inv = 1.0f / (modulo + eps);

            // normalize
            if (channel_shared) {
                float fused_weight_modulo = weights[0] * modulo_inv;
                parallel_for2d(H, W, [&](int ih, int iw) {
                    const in_data_t *src_data_bhw = src_data_b + ih * C * W + iw * C;
                    out_data_t *dst_data_bhw = dst_data_b + ih * C * W + iw * C;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_bhw;
                    arg.dst = dst_data_bhw;
                    arg.fused_factor = static_cast<float*>(&fused_weight_modulo);  // bc static
                    arg.oc_off = 0;
                    arg.work_amount = static_cast<size_t>(C);
                    (*normalize_kernel)(&arg);
                });
            } else {  // channel_shared=false
                std::vector<float> fused_weight_modulo(C);
                for (size_t c = 0; c < C; c++) {
                    fused_weight_modulo[c] = weights[c] * modulo_inv;
                }
                parallel_for2d(H, W, [&](int ih, int iw) {
                    const in_data_t *src_data_bhw = src_data_b + ih * C * W + iw * C;
                    out_data_t *dst_data_bhw = dst_data_b + ih * C * W + iw * C;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_bhw;
                    arg.dst = dst_data_bhw;
                    arg.fused_factor = static_cast<float *>(&fused_weight_modulo[0]);  // ld dynamic
                    arg.oc_off = 0;
                    arg.work_amount = static_cast<size_t>(C);
                    (*normalize_kernel)(&arg);
                });
            }
        } else {  // for across_spatial=false
            parallel_for2d(H, W, [&](int ih, int iw) {
                // modulo
                float modulo = 0.f;
                const in_data_t *src_data_bhw = src_data_b + ih * C * W + iw * C;
                out_data_t *dst_data_bhw = dst_data_b + ih * C * W + iw * C;
                auto arg = jit_normalize_call_args();
                arg.src = src_data_bhw;
                arg.modulo = static_cast<float*>(&modulo);
                arg.src_stride = blk_size * sizeof(in_data_t);
                arg.work_amount = C / blk_size;
                (*normalize_modulo_kernel)(&arg);

                size_t tail_start = (C / blk_size) * blk_size;

                // for tail
                for (size_t c = tail_start; c < C; c++) {
                    modulo += src_data_bhw[c] * src_data_bhw[c];
                }

                modulo = std::sqrt(modulo);
                float modulo_inv = 1.0f / (modulo + eps);

                // normalize
                arg.dst = dst_data_bhw;
                float fused_weight_modulo = 0;
                if (channel_shared) {
                    fused_weight_modulo = modulo_inv * weights[0];
                    arg.fused_factor = static_cast<float*>(&fused_weight_modulo);  // bc static
                } else {
                    arg.modulo = static_cast<float*>(&modulo_inv);  // bc static
                    arg.weights = static_cast<float*>(&weights[0]); // ld dynamic
                }
                arg.work_amount = C;
                arg.oc_off = 0;
                (*normalize_kernel)(&arg);
            });
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeNode::normalize_blk(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims) {
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(cpu::avx512_common)) {
        blk_size = 16;
    } else if (mayiuse(cpu::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::sse42)) {
        blk_size = 8;
    }

    size_t dims_size = dims.size();
    size_t W = (dims_size > 3) ? dims[3] : 1lu;
    size_t H = (dims_size > 2) ? dims[2] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t B = (dims_size > 0) ? dims[0] : 1lu;
    float *weights = weights_blob->buffer().as<float *>();

    size_t CB = div_up(C, blk_size);

    // normalize for tails: data is padding, norm weight is padding, so tails as vector for normalize;
    // post ops for tails: post-ops params is padding.
    std::vector<float> weights_padding(CB * blk_size);
    if (!channel_shared) {
        memcpy(static_cast<float*>(&weights_padding[0]), weights, C * sizeof(float));
    }

    for (size_t b = 0lu; b < B; b++) {
        const in_data_t *src_data_b = src_data + b * CB * H * W * blk_size;
        out_data_t *dst_data_b = dst_data + b * CB * H * W * blk_size;
        if (across_spatial) {
            // modulo
            float modulo = 0.0f;
            float addition_identity = 0.0f;
            modulo = parallel_sum2d(CB, H, addition_identity, [&](size_t cb, size_t h) -> float {
                // handle W * blk_size data
                const in_data_t *src_data_b_cb_h = src_data_b + cb * H * W * blk_size + h * W * blk_size;
                size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                float modulo_w_blk = 0.0f;
                if (min_cb == blk_size) {
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_b_cb_h;
                    arg.modulo = static_cast<float*>(&modulo_w_blk);
                    arg.src_stride = blk_size * sizeof(in_data_t);
                    arg.work_amount = W;
                    (*normalize_modulo_kernel)(&arg);
                } else {
                    for (size_t w = 0; w < W; w++) {
                        const in_data_t *src_data_b_cb_h_w = src_data_b_cb_h + w * blk_size;
                        for (size_t c = 0; c < min_cb; c++) {
                            modulo_w_blk += src_data_b_cb_h_w[c] * src_data_b_cb_h_w[c];
                        }
                    }
                }
                return modulo_w_blk;
            });

            modulo = std::sqrt(modulo);
            float modulo_inv = 1.0f / (modulo + eps);

            // normalize
            if (channel_shared) {
                float fused_weight_modulo = weights[0] * modulo_inv;
                parallel_for2d(CB, H, [&](size_t cb, size_t h) {
                    const in_data_t *src_data_b_cb_h = src_data_b + cb * H * W * blk_size + h * W * blk_size;
                    out_data_t *dst_data_b_cb_h = dst_data_b + cb * H * W * blk_size + h * W * blk_size;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_b_cb_h;
                    arg.dst = dst_data_b_cb_h;
                    arg.fused_factor = static_cast<float*>(&fused_weight_modulo);  // broadcast once
                    arg.work_amount = static_cast<size_t>(W);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    (*normalize_kernel)(&arg);
                });
            } else {
                for (size_t c = 0; c < C; c++) {
                    weights_padding[c] = weights_padding[c] * modulo_inv;
                }
                parallel_for2d(CB, H, [&](size_t cb, size_t h) {
                    const in_data_t *src_data_b_cb_h = src_data_b + cb * H * W * blk_size + h * W * blk_size;
                    out_data_t *dst_data_b_cb_h = dst_data_b + cb * H * W * blk_size + h * W * blk_size;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_b_cb_h;
                    arg.dst = dst_data_b_cb_h;
                    arg.fused_factor = static_cast<float*>(&weights_padding[cb * blk_size]);  // load once
                    arg.work_amount = static_cast<size_t>(W);
                    arg.oc_off = cb * blk_size  * sizeof(float);
                    (*normalize_kernel)(&arg);
                });
            }
        } else {  // across_spatial: false
            parallel_for2d(H, W, [&](size_t ih, size_t iw) {
                // modulo
                float modulo = 0.0f;
                const in_data_t *src_data_bhw = src_data_b + ih * W * blk_size + iw * blk_size;
                out_data_t *dst_data_bhw = dst_data_b + ih * W * blk_size + iw * blk_size;
                auto arg = jit_normalize_call_args();
                arg.src = src_data_bhw;
                arg.modulo = static_cast<float*>(&modulo);
                arg.src_stride = blk_size * W * H * sizeof(in_data_t);
                arg.work_amount = C / blk_size;  // CB or CB-1
                (*normalize_modulo_kernel)(&arg);
                // for tail
                size_t padding = CB * blk_size - C;
                if (padding > 0) {
                    size_t tail = blk_size - padding;
                    const in_data_t *src_data_bhw_lastCB = src_data_bhw + (CB - 1) * blk_size * W * H;
                    for (size_t c = 0; c < tail; c++) {
                        modulo += src_data_bhw_lastCB[c] * src_data_bhw_lastCB[c];
                    }
                }

                modulo = std::sqrt(modulo);
                float modulo_inv = 1.0f / (modulo + eps);

                // normalize
                arg.dst = dst_data_bhw;
                float fused_weight_modulo = 0;
                if (channel_shared) {
                    fused_weight_modulo = weights[0] * modulo_inv;
                    arg.fused_factor = static_cast<float*>(&fused_weight_modulo);  // broadcast
                } else {
                    arg.weights = static_cast<float*>(&weights_padding[0]);  // load
                    arg.modulo = static_cast<float*>(&modulo_inv);  // broadcast
                }
                arg.work_amount = CB;
                arg.oc_off = 0;
                (*normalize_kernel)(&arg);
            });
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeNode::normalize_function(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims) {
    auto selectedPD = getSelectedPrimitiveDescriptor();
    Layout selected_layout = selectedPD->getConfig().inConfs[0].desc.getLayout();
    if (mayiuse(cpu::sse42) && normalize_modulo_kernel && normalize_kernel) {
        if (selected_layout == MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims())) {
            normalize_nchw(src_data, dst_data, dims);
        } else if (selected_layout == Layout::NHWC) {
            normalize_nhwc(src_data, dst_data, dims);
        } else if (selected_layout == Layout::BLOCKED) {
            normalize_blk(src_data, dst_data, dims);
        } else {
            THROW_IE_EXCEPTION << "The selected layout is not supported.";
        }
    } else {
        if (selected_layout == MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims())) {
            normalize_nchw_ref(src_data, dst_data, dims);
        } else {
            THROW_IE_EXCEPTION << "Only support plain layout on machine w/o sse42.";
        }
    }
}

inline void MKLDNNNormalizeNode::apply_post_ops_scalar(float &dst_value, int index_c) {
    const auto &p = (*attr.get()).post_ops_;
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    for (int i = 0; i < p.len_; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            dst_value = eltwise_injectors_ref[eltwise_inj_idx]->compute_scalar(dst_value);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            auto depthwise_weights = post_op.depthwise.weights_data + index_c;
            auto depthwise_bias = post_op.depthwise.biases_data + index_c;
            dst_value = depthwise_injectors_ref[depthwise_inj_idx]->compute_scalar(dst_value, depthwise_weights, depthwise_bias);
            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || output_prec == Precision::FP32 || i != p.len_ - 1;

            auto quant = post_op.quantization;

            float crop_low = quant.crop_low_data->shifts_[quant.crop_low_data->count_ == 1 ? 0 : index_c];
            float crop_high = quant.crop_high_data->shifts_[quant.crop_high_data->count_ == 1 ? 0 : index_c];
            float input_scale = quant.input_scale_data->scales_[quant.input_scale_data->count_ == 1 ? 0 : index_c];
            float input_shift = quant.input_shift_data->shifts_[quant.input_shift_data->count_ == 1 ? 0 : index_c];

            dst_value = nstl::min(crop_high, nstl::max(crop_low, dst_value));
            dst_value = dst_value * input_scale + input_shift;

            if (do_rounding) {
                dst_value = roundf(dst_value);
            }

            if (do_dequantization) {
                float output_scale = quant.output_scale_data->scales_[quant.output_scale_data->count_ == 1 ? 0 : index_c];
                float output_shift = quant.output_shift_data->shifts_[quant.output_shift_data->count_ == 1 ? 0 : index_c];
                dst_value = dst_value * output_scale + output_shift;
            }
        }
    }
}

bool MKLDNNNormalizeNode::created() const {
    return getType() == Normalize;
}

REG_MKLDNN_PRIM_FOR(MKLDNNNormalizeNode, Normalize);
